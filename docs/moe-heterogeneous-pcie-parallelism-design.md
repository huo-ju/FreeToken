# 异构 PCIe 环境下的 MoE 并行与专家执行设计

> 状态：设计讨论稿，尚未开始实现
> 记录日期：2026-08-24
> 目标平台：4 × NVIDIA TITAN RTX（SM75）单机，DeepSeek-V4-Flash-0731
> 相关文档：[MoE 多层资源管理与弹性 Placement 改造计划](moe-resource-placement-plan.md)

## 1. 摘要

当前系统已经能够在 4 张 TITAN RTX 上正确加载并通过 OpenAI 兼容 API
运行 DeepSeek-V4-Flash-0731，但 decode 性能约为 4.7 token/s，GPU 功耗和显存
带宽压力都不高。实测显示主要限制来自异构 PCIe 拓扑：GPU3 只有 PCIe Gen3 x4，
而当前 routed MoE 使用对称 TP4。每个命中的专家都必须由四个 rank 各自加载并计算
四分之一权重，随后逐层 `all_reduce`；因此最慢的 x4 rank 会成为每个 MoE 层的
同步瓶颈。

本设计讨论以下缓解路线：

1. 扩大 dynamic cache、增加 permanent residency、恢复 prefill overlap；
2. 为不同 rank 配置不等宽 routed-expert tensor parallelism；
3. 静态或动态 Expert Parallel（EP）；
4. 只让 GPU3 执行 resident experts，避免运行期频繁经过 x4 加载权重；
5. 最终采用混合执行：
   - resident experts 使用 EP；
   - cold miss 根据实时状态选择带宽感知 TP、EP 或 CPU；
   - 在同一层内重叠 cache-hit 计算、shared expert、H2D copy、GPU expert
     计算和 CPU expert 计算。

最重要的结论是：**EP 本身不会减少一个 cache miss 所需的总权重字节数，也不会
自动隐藏传输。** 对单请求、小 batch、每层只激活 6 个专家的 decode，纯 EP 还会
失去把单个专家分散到多条 PCIe 链路并行传输的能力。当前平台更有潜力的长期方案
不是固定 TP 或固定 EP，而是以每个 `(layer, expert)` 的 residency、链路带宽、队列和
cache 状态为输入，动态选择执行形态。

## 2. 当前系统背景

### 2.1 硬件

当前机器的相关资源如下：

| 资源 | 状态 |
|---|---|
| GPU | 4 × NVIDIA TITAN RTX，SM75，每张 24 GiB VRAM |
| GPU0 | PCIe Gen3 x8（设备最大 x16） |
| GPU1 | PCIe Gen3 x8（设备最大 x16） |
| GPU2 | PCIe Gen3 x16 |
| GPU3 | PCIe Gen3 x4 |
| GPU 间拓扑 | 同一 NUMA node，经 PHB/NODE 连接 |
| NVLink | 当前没有 |
| CPU/NUMA | 单 NUMA node，CPU affinity 0-23 |
| 系统内存 | 约 123 GiB |
| Swap | 2 GiB；模型运行时 worker `VmSwap=0`，系统有约 91 MiB 历史占用 |

PCIe lane 数受主板和 CPU 的物理限制，短期内不能调整。未来可增加 NVLink，但
NVLink 只会自动帮助 GPU 间 collective；当前 host expert shard 仍然直接从 CPU RAM
复制到目标 GPU。若要通过 NVLink 绕开 GPU3 x4，还需要显式实现
`CPU -> fast-link GPU -> NVLink -> GPU3` 的 relay 路径。

### 2.2 模型几何

DeepSeek-V4-Flash-0731 的关键参数：

| 参数 | 值 |
|---|---:|
| hidden size | 4096 |
| MoE layers | 43 |
| routed experts / layer | 256 |
| experts activated / token | 6 |
| shared experts | 1 |
| routed expert intermediate size | 2048 |
| hash-routing layers | 3 |
| routed expert dtype | FP4 |

当前 TP4 下：

- 每个 rank 的 routed expert intermediate shard 为 512；
- 每个 rank 的一个 expert shard 为 3,342,336 bytes；
- 一个未分片完整 expert 约为 13,369,344 bytes（约 12.75 MiB）；
- 一个完整 expert layer 的 aggregate payload 为 3.1875 GiB；
- 一个 rank 上一个完整 layer 的 TP4 shard 为 0.796875 GiB；
- 43 层完整 host expert pool 约为 137 GiB。

由于系统只有约 123 GiB RAM，不能沿用论文中“全部 routed experts 常驻 host RAM”
的单 GPU 布局。当前 placement 将 16 层 routed experts 放入 GPU permanent store：

```text
host expert backing
  = (43 - 16) × 3.1875 GiB
  ≈ 86.1 GiB
```

因此 `--moe-host-budget-gb 88` 可以成立；其余 RAM 留给 non-expert host state、
进程、页缓存和运行时。

### 2.3 当前服务基线

最近用于真实模型验证的主要配置为：

```bash
python -m freetoken.cli serve \
    --model /data/models/DeepSeek-V4-Flash-0731 \
    --tp-size 4 \
    --moe-backend offload \
    --nvfp4-backend triton \
    --moe-placement auto \
    --moe-host-budget-gb 88 \
    --moe-pin-budget-gb auto \
    --moe-placement-policy balanced \
    --moe-cache-size 256 \
    --moe-gpu-only-layers 16 \
    --expert-load serial \
    --disable-moe-prefill-overlap \
    --max-running-requests 1 \
    --cuda-graph-max-bs 0 \
    --max-seq-len-override 16384 \
    --num-tokens 16384 \
    --memory-ratio 1.0 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port 1919
```

代表性 benchmark：

```bash
vllm bench serve \
    --input-len 12000 \
    --output-len 1024 \
    --num-prompts 100 \
    --num-warmups 5 \
    --request-rate inf \
    --max-concurrency 1
```

这是一组单请求延迟测试，不是服务吞吐测试。`max-concurrency=1` 会减少并行可调度
的 token 数量，也会放大每层 cache miss 和同步延迟。

### 2.4 已观测性能与资源状态

最近一次有效观测：

| 指标 | 观测值 |
|---|---:|
| decode throughput | 约 4.7 token/s |
| mean TTFT | 约 41.4 s |
| GPU 显存 | 每张约 23,210 MiB |
| GPU SM utilization | 采样中约 67%–99% |
| GPU memory utilization | 约 8%–9% |
| GPU power | 约 102–123 W / 280 W limit |
| worker swap | 0 |

10 秒 `nvidia-smi dmon` 采样中的 host-to-device 流量大致为：

| GPU | 观测 H2D / PCIe RX 峰值或常见值 |
|---|---:|
| GPU0 | 约 4.8 GB/s |
| GPU1 | 约 4.8 GB/s |
| GPU2 | 约 6.1 GB/s |
| GPU3 | 多数约 2.5–2.9 GB/s |

CPU 没有饱和，模型 worker 没有实际使用 swap。低 GPU memory utilization、较低功耗、
明显的 H2D 流量以及 GPU3 的低带宽共同表明：当前主要限制是 host-backed routed
expert streaming，而不是 HBM 容量、系统 swap 或纯 GPU 算力。

这些数字是一次具体运行的诊断基线，不应当视为稳定硬件常数。后续 planner 必须使用
并发传输条件下的实测带宽，而不是仅使用 PCIe lane 标称值。

## 3. FreeToken 论文与当前多 GPU 实现的边界

FreeToken 论文研究的是单 GPU edge serving。论文中的核心布局是：

- CPU-resident expert pool 保存完整 routed experts，并作为 source of truth；
- non-expert weights 常驻 GPU；
- 剩余 VRAM 是跨层共享的完整专家 LRU cache；
- prefill 使用整层双缓冲，将下一层传输与当前层计算重叠；
- decode 将 miss 分成 GPU cache fill 和 CPU direct execution 两支，并由 `q*`
  根据实测 host/PCIe 带宽进行平衡；
- CPU 和 GPU 计算 exact partial sums，最后合并。

论文没有讨论 tensor parallel、expert parallel、multi-GPU 或 all-to-all。论文列出的
3090/4090/5090 等系统都是单 GPU 实验机器；DSV4 的端到端结果主要来自 32 GiB
RTX 5090，并依赖 180–192 GiB host RAM 保存约 140 GB expert pool。

当前 DSV4 TP4 是后续多 GPU 扩展：每个 rank 保留所有 expert IDs，但只保存和计算
每个专家 intermediate 维度的四分之一。论文中的“complete expert slot”在当前实现中
被泛化为“四个 rank-local shards 共同构成一个逻辑 expert slot”。

因此，本文件讨论的异构 TP、EP 和混合执行都是论文思想向多 GPU 异构互连环境的
扩展，不是论文已经给出的设计。

## 4. 当前执行路径与瓶颈

### 4.1 对称 TP4

当前每个 routed expert 沿 SwiGLU intermediate 维均分：

```text
Expert e, intermediate=2048
  GPU0: [   0,  512)
  GPU1: [ 512, 1024)
  GPU2: [1024, 1536)
  GPU3: [1536, 2048)
```

每个 rank 都运行相同 router，得到相同 expert IDs。对于每个 cache miss：

1. rank 从自己的 host bank 加载该 expert 的本地 shard；
2. rank 计算本地 partial output；
3. routed partial 与 shared-expert partial 在本地相加；
4. 四个 rank 对 `[tokens, hidden_size]` 做一次 `all_reduce`。

一个层的关键路径可以粗略表示为：

```text
T_layer ~= max_i(
    copy_bytes_i / effective_H2D_i
    + local_expert_compute_i
) + T_all_reduce
```

对称 TP 令每个 rank 的 `copy_bytes_i` 相等，因此 GPU3 x4 的传输时间最大。逐层
`all_reduce` 又要求所有 rank 等待它完成，最终使 x4 链路锁住整条 decode 链路。

改变 rank 顺序、把 GPU3 放在最后、调整 NCCL ring 或进程启动顺序，都不会减少
GPU3 必须加载的 expert shard 字节数，因此不能消除这个瓶颈。

### 4.2 Prefill

论文的 prefill 路径通过整层双缓冲实现：

```text
compute layer L      ───────────────>
copy layer L+1          ───────────────>
```

理想情况下：

```text
T_layer ~= max(T_compute_current, T_copy_next)
```

如果 GPU3 的四分之一 layer shard 传输时间超过当前层计算时间，超出的部分仍然会暴露。
此外，当前基线显式使用 `--disable-moe-prefill-overlap`，所以目前长 prompt 测试尚未
使用论文的主要 prefill 隐藏机制。任何新并行架构的 prefill 对比都必须同时记录
overlap on/off，不能将禁用 overlap 的基线归因于并行方式本身。

### 4.3 Decode

除前三个 hash-routing layers 外，下一层的 expert IDs 依赖当前层 hidden state，不能
在一般情况下准确提前获知。因此 decode 不能像 prefill 那样提前加载整个下一层而
保持低流量。

Decode 可隐藏的工作主要局限在同一层内：

- shared expert 计算；
- resident/cache-hit expert 计算；
- 其他 GPU 的 copy 或 expert 计算；
- 同一 GPU 上前一 expert 的计算；
- CPU direct-execution branch；
- 少数 hash-routing layers 的模型特定预取。

最后一个当前层必需的 cache miss 无法隐藏在下一层计算后面，因为下一层必须等待本层
exact output。

## 5. 候选方案

### 5.1 恢复 prefill overlap

最先验证的低风险改动是删除：

```text
--disable-moe-prefill-overlap
```

该方案主要改善 TTFT，不解决 decode 中逐层按需 miss。它还需要足够的 GPU slot pool
容纳两个 full-layer streaming buffers；否则会回退到 on-demand prefill。

### 5.2 扩大 dynamic cache

将 `--moe-cache-size` 从 256 提高到 512 等值可以降低 decode miss rate。每个 rank
每增加 256 个当前 TP4 expert slots，约增加 816 MiB VRAM。当前显存已经接近 24 GiB，
需要与 KV cache、CUDA Graph pool 和 permanent store 一起验证，不能仅凭静态估算。

如果实现 per-rank cache budgets，可以给 GPU3 更高的 cache 命中目标，从而降低其
x4 传输。不过 GPU3 并没有额外显存，增加 dynamic cache 必须减少其他池，且可能只将
容量问题转移到 KV cache。

### 5.3 非对称 permanent placement

当前 permanent tier 以整层、所有 rank 对称的方式放置。可以允许 GPU3 对更多 layer
或 expert shard 使用 permanent residency，让其在 decode 中少做 H2D fill；其他 rank
仍然动态加载自己的 partial shards。

优点：

- 不改变数学分片和 collective；
- 直接减少 GPU3 的运行期传输；
- startup 的一次性 x4 传输可以摊销。

局限：

- GPU3 VRAM 已接近满载；
- 每增加一个完整 layer 的 GPU3 shard 约需 0.796875 GiB；
- 只永久增加一层仅减少约 `1 / 27` 的 host-backed layer traffic；
- 按 expert 而非按 layer permanent 更灵活，但需要把 planner 和 store 粒度下沉到
  `(layer, expert)`。

### 5.4 MoE-only 非均匀 TP

保持 attention、dense 和 shared expert 的现有 TP4，仅改变 routed experts 的
intermediate shard 宽度。

DSV4 intermediate size 为 2048，当前 SM75 decode kernel 要求本地宽度按 256 对齐，
因此一共有 8 个分配单位。第一组候选为：

```text
GPU0: 512  (2 units, 25.0%)
GPU1: 512  (2 units, 25.0%)
GPU2: 768  (3 units, 37.5%)
GPU3: 256  (1 unit, 12.5%)
```

即 `2:2:3:1`。

按照当前独立 H2D 观测值，纯带宽理想比例约为：

```text
4.8 : 4.8 : 6.1 : 2.8
~= 26% : 26% : 33% : 15%
```

`2:2:3:1` 是满足 256 对齐约束的近似，而不是理论最优。GPU2 同时需要承担 50%
更多 expert compute，因此最终比例必须通过真实 kernel profile 校准。

优点：

- 一个 expert 可以被多条 PCIe 链路同时条带化加载；
- 单个或少量 miss 时仍能利用多 GPU；
- 每层负载确定，不受 top-k expert ID 分布影响；
- 保持一次 output all-reduce；
- 不需要 token all-to-all。

代价：

- checkpoint loader、host bank、permanent store、cache slot 和 kernel shape 都要支持
  per-rank width/offset；
- 当前大量 `div_even()`、相同 slot bytes 和对称预算假设需要解除；
- GPU2 的传输和计算可能成为新瓶颈；
- topology-specific 配置不一定适合直接提交上游。

### 5.5 静态 Expert Parallel

静态 EP 为每个 `(layer, expert)` 指定一个 owner GPU。owner 保存和计算完整专家；其他
rank 对该 expert 输出零，最后仍通过 `[T,H] all-reduce` 合并。

当前 TP runtime 的 hidden state 和 router 结果在各 rank 上复制，因此第一版 EP 不必
实现标准数据中心 EP 的 token all-to-all：所有 rank 都已经知道输入和 routes，只需
过滤本 rank owned routes。

优点：

- GPU3 可以拥有更少 offloaded experts；
- 一个 expert miss 只在一个 rank 上发生；
- 完整 intermediate=2048 kernel 可能比 512-wide 小 shard kernel 更高效；
- current output all-reduce 可以复用。

局限：

- 一个完整 miss 必须经过 owner 的一条 PCIe 链路，不能条带化；
- top-k=6、batch=1 时专家任务数量少，owner collision 会形成尾延迟；
- 静态 ownership 很难适应不断变化的 route distribution；
- cache slot 从约 3.19 MiB/rank shard 变为约 12.75 MiB full expert；aggregate cache
  容量不变，但每 GPU 的任务和容量粒度更粗；
- host banks、cache lookup、prefill streaming 和 permanent placement 都要理解全局
  expert ID 到 owner/local ID 的映射。

EP 不减少总传输量。若 `m` 个完整 experts 都 miss：

```text
TP aggregate H2D bytes = m × S
EP aggregate H2D bytes = m × S
```

区别仅在于 TP 将每个 `S` 分散到多条链路，而 EP 将完整 `S` 放到一个 owner。

### 5.6 动态 EP / Expert Task Parallelism

相比固定 owner，更符合 FreeToken 思路的是：host expert pool 保持 authoritative，
每个 cache miss 在运行时选择预计最早完成的 GPU。cache hit 则继续在已有 resident GPU
执行，避免迁移。

调度输入至少包括：

- 每张 GPU 的实测同时 H2D 带宽；
- 当前 copy queue 和 compute queue；
- expert 是否已经 resident；
- cache victim 的未来价值和迁移成本；
- GPU kernel 对不同 batch/route count 的实测成本；
- aggregate host-memory bandwidth；
- CPU branch 当前负载；
- all-reduce 固定成本。

一个简化的 owner 选择目标为：

```text
predicted_finish_i
  = queued_copy_bytes_i / Bp_i
  + new_copy_bytes_i / Bp_i
  + queued_compute_i
  + expert_compute_i
  + eviction_penalty_i
```

选择最小 `predicted_finish_i` 的 GPU 只是第一步；同一层有多个 misses 时，应联合求解
最小化所有 rank 的最大完成时间。

主要难点是 host source 可见性。当前每个 rank 只持有自己的 TP shard。任意 GPU 若要
动态加载完整 expert，不能让四个进程各自复制一份 137 GiB full host pool。可选实现有：

1. 单份 shared-memory/file-backed host bank，由多个 rank 映射并注册；
2. 独立 transfer broker 持有 authoritative banks，通过 CUDA IPC 向目标 GPU DMA；
3. host banks 按 static owner 分区，只允许 owner 加载其 full experts；
4. 冷专家继续使用 TP-sharded host source，只有 permanent/resident experts 使用 EP；
5. 将 FTW 格式扩展为可按完整 expert 直接访问的共享只读布局。

必须验证多进程对同一物理 host pages 的 pin/register 行为和 pin quota 记账，不能假设
共享文件映射天然满足高速 DMA 与内存不重复占用。

### 5.7 Pipeline Parallel

可以让 GPU3 只负责少量层，其他 GPU 负责更多层，从而减少 GPU3 的 expert traffic。
但对 `max-concurrency=1`，pipeline stages 基本串行，失去 TP4 的层内计算并行。只有在
足够多 microbatches 或并发请求填满 pipeline 时才可能改善吞吐。

当前目标是单请求 agentic latency，因此 PP 不作为近期主路线。未来如果服务目标转向
高并发吞吐，可重新评估不均匀 PP 或 PP × TP hybrid。

### 5.8 TP2、TP3 或排除 GPU3

- TP3 受 DSV4 intermediate/head shape、256 tile 和现有等分假设限制，不能直接启用；
- TP2 会显著提高每卡常驻 non-expert、expert buffer 和 cache 占用，同时改变 host RAM
  与 permanent placement 的可行解；
- 仅调整 `CUDA_VISIBLE_DEVICES` 排除 GPU3 不能保证当前 16K 配置可以装入并运行。

这些方案需要重新做完整 capacity planning，不能当成零代码 workaround。

### 5.9 NVLink 与 P2P relay

安装 NVLink 后，NCCL 可能自动使用它加速支持的 collective，但当前 H2D expert fill
仍然直接走目标 GPU 的 PCIe。要让 GPU3 绕开 x4，需要显式路径：

```text
host pinned bank
      |
      | PCIe x16
      v
GPU2 staging slot
      |
      | NVLink P2P
      v
GPU3 expert slot
```

这需要：

- GPU2 staging capacity；
- P2P copy stream 和跨设备 event；
- relay 与本地 copy 的 cost model；
- cache slot 生命周期和失败恢复；
- 实际 NVLink pairing/topology 验证。

NVLink relay 可以作为未来动态调度器的一种 transport，不能替代调度器本身。

## 6. 方案对比

| 方案 | 减少 GPU3 H2D | 单 miss 利用多链路 | 负载粒度 | Decode overlap 潜力 | 实现成本 |
|---|---|---|---|---|---|
| 增大 cache | 间接 | 保持当前 TP | expert slot | 低到中 | 低 |
| 非对称 permanent | 是 | 保持当前 TP | layer/expert shard | 低 | 中 |
| `2:2:3:1` 非均匀 TP | 是 | 是 | 256-wide shard | 中 | 中到高 |
| 静态 EP | 是 | 否 | 完整 expert | 中 | 高 |
| 动态 EP | 是 | 否 | 完整 expert | 中到高 | 很高 |
| Pipeline Parallel | 是 | 不适用 | layer | 高并发时高 | 很高 |
| NVLink relay | 可绕开 x4 | 取决于计划 | shard/expert | 中到高 | 高 |
| 动态 TP/EP/CPU hybrid | 是 | 可选择 | shard 或 expert | 最高 | 最高 |

## 7. 推荐长期架构：Resident EP + Cold Dynamic TP/EP

### 7.1 核心原则

不为所有 experts 固定一种并行方式，而是依据状态选择：

```text
resident hit
  -> 在 resident GPU 上以完整专家 EP 执行

cold miss，数量少或单个 miss
  -> 在多条链路上做 bandwidth-weighted TP

cold miss，数量足够并行
  -> 将完整 experts 作为任务动态分配给多个 GPU（EP）

host bandwidth 有剩余、CPU 分支预计更快
  -> CPU direct execution
```

目标不是最大化某一张卡的吞吐，而是最小化当前 MoE 层所有必需 partial outputs 的
最大完成时间。

### 7.2 建议的 residency tiers

```text
Tier A: EP_PERMANENT
  - 完整 expert 常驻一张 GPU
  - host backing 可释放
  - 运行期不发生 H2D fill

Tier B: EP_DYNAMIC
  - 完整 expert 当前缓存于一张 GPU
  - host source 保留
  - hit 在 owner 执行，eviction 后可更换 owner

Tier C: TP_DYNAMIC
  - authoritative host expert 按 rank 保存 weighted shards
  - miss 时多 GPU 并行加载和计算

Tier D: CPU_DIRECT
  - expert 保持 host resident
  - CPU 直接计算 exact partial output
```

同一个 `(layer, expert)` 在某一时刻只能有一个 authoritative representation 组合，
planner 必须明确其 source、resident replicas 和允许的执行模式，防止 host pages 已释放
后仍被当成 dynamic source。

### 7.3 GPU3 的角色

GPU3 建议优先承担：

- 启动时一次性加载的 EP permanent experts；
- 长期 hot、迁移收益高的 resident experts；
- attention/dense 的现有 TP4 shard；
- output all-reduce；
- 只有在预测完成时间有利时才承担 cold fill。

GPU3 不应默认承担与其他 rank 相同的 cold miss 字节数。其 x4 链路是运行期动态换入的
高成本资源，但 GPU3 的 HBM 和 SM 仍然是可用资源。

### 7.4 GPU0/1/2 的角色

- GPU2 优先承担更高比例的 cold H2D traffic；
- GPU0/1 承担中等比例；
- 多个完整 EP misses 应按队列和实测吞吐分散；
- 单个 miss 优先使用 weighted TP 条带化，而不是强制放到一个 owner；
- 不允许仅按 PCIe lane 数推导比例，必须使用真实 tensor shape 下的并发 benchmark。

### 7.5 同层执行流水

Router 完成后：

1. 在设备端去重 routed expert IDs；
2. 查询全局或一致的 distributed residency directory；
3. resident hits 立即进入各 GPU compute queue；
4. 启动 shared expert；
5. 对 misses 求解 EP/weighted-TP/CPU 分配；
6. 每张 GPU 使用独立 copy stream 填充目标 slot；
7. copy A 完成后计算 A，同时 copy B；
8. CPU workers 并发计算 CPU-assigned experts；
9. 将本 rank 所有 routed/shared partials累加到固定形状 `[T,H]` buffer；
10. 一次 all-reduce；
11. 更新 residency、LRU 和统计信息。

理想化关键路径从：

```text
T_copy + T_compute
```

接近变为：

```text
max(T_copy, T_compute, T_cpu) + pipeline_startup_tail
```

但 decode 末尾必需 miss 的 tail 无法完全隐藏。

### 7.6 为什么保留 weighted TP

以 6 个全部 miss 的专家为例，纯传输近似：

```text
当前 equal TP:
  每个 rank 约 6 × 12.75 MiB / 4 = 19.1 MiB
  GPU3 关键时间约由 19.1 MiB / 2.8 GB/s 决定

2:2:3:1 weighted TP:
  GPU0 ≈ 19.1 MiB
  GPU1 ≈ 19.1 MiB
  GPU2 ≈ 28.7 MiB
  GPU3 ≈  9.6 MiB

dynamic EP 的一种整数分配:
  GPU0: 2 full experts
  GPU1: 2 full experts
  GPU2: 1 full expert
  GPU3: 1 full expert
```

Dynamic EP 可以明显好于当前 equal TP，但完整 expert 是粗粒度任务。只有一个 miss 时，
EP 必须通过单一链路加载约 12.75 MiB，而 weighted TP 可以将它同时分散到多条链路。
因此纯 EP 并不是单流 decode 的普遍最优解。

### 7.7 Cache 与 ownership

Distributed cache directory 至少需要表达：

```text
(layer_id, expert_id) -> {
    source_tier,
    host_layout,
    resident_gpu,
    resident_slot,
    replica_mask,
    last_access,
    in_flight_state,
    allowed_execution_modes,
}
```

需要解决：

- 同一 expert 是否允许多个 GPU replica；
- hot expert replication 是否值得消耗 aggregate cache capacity；
- eviction 是否需要跨 rank 一致提交；
- in-flight copy 时新的 route 如何复用同一任务；
- permanent expert 如何从 dynamic directory 中排除；
- CUDA Graph replay 中如何表达动态 owner 和有效任务数；
- runtime cache rebuild 时如何保留仍有价值的 distributed entries。

### 7.8 全局 host-bandwidth 约束

四张 GPU 的 H2D DMA 和 CPU direct execution 都读取同一套主存。不能对每个 rank 独立
应用论文单 GPU 的 `q*`，否则四个 rank 可能同时认为自己拥有全部 host bandwidth。

Multi-GPU planner 应使用：

- 并发 H2D 时每张 GPU 的有效带宽；
- 所有 H2D 的 aggregate DRAM read bandwidth；
- CPU expert kernel 与 DMA 同时运行时的 residual bandwidth；
- NUMA 和 memory-controller contention；
- pinned、locked 和 pageable source 的不同带宽。

最终的 `q*` 应扩展为全机联合分配问题，而不是四个独立标量。

## 8. 传输成本能够隐藏到什么程度

### 8.1 可以隐藏的部分

- prefill 的下一层完整 expert partition；
- decode 中 misses 的 copy 与 shared expert；
- misses 的 copy 与 resident-hit expert compute；
- 同 GPU 上 expert B 的 copy 与 expert A 的 compute；
- 不同 GPU 的 copy/compute；
- GPU fill branch 与 CPU direct branch；
- hash-routing layers 的有限预取；
- 使用 NVLink 后的 P2P relay 与其他 GPU compute。

### 8.2 无法一般性隐藏的部分

- 当前层最后一个必需 cache miss 的完成 tail；
- router 完成之前未知的 scored-router expert IDs；
- host DRAM 已饱和时额外 DMA 或 CPU execution；
- cache eviction/ownership 需要的全局同步；
- 最终 output collective 的延迟；
- batch=1、miss 数量很少时不足以填满多个 expert queues 的空洞。

### 8.3 预期判断

相对当前 equal TP4，weighted TP 或动态 EP 都有较高概率改善性能，因为它们直接减少
GPU3 的关键路径负载。

相对 `2:2:3:1` weighted TP，纯 EP 不保证更快：

- cold miss 很少时，weighted TP 的链路条带化更有利；
- misses 多、cache hits 分散、完整 expert kernel 效率更高时，EP 更可能获益；
- resident EP 不需要传输，最具确定性价值；
- 动态混合策略只有在调度开销足够低并保留 CUDA Graph 能力时才可能持续获益。

任何性能目标都必须以最佳 weighted-TP baseline 为对照，而不能只和当前 equal TP 比较。

## 9. 建议实施路线

### Phase 0：冻结诊断基线

- 固定模型、prompt、输出长度、cache 初始状态和并发度；
- 分别测 prefill overlap on/off；
- 记录每层每 rank 的 H2D bytes/time、expert compute、all-reduce 和 cache misses；
- 测量四 GPU 同时 DMA 时的 effective bandwidth；
- 测量不同 intermediate width 的 SM75 expert kernel；
- 区分 cold start、warm cache 和稳定 agent trace。

### Phase 1：MoE-only weighted TP

- 增加 per-rank routed-expert shard width/offset；
- 第一组实现 `512,512,768,256`；
- generalized host-bank specs、loader、cache slot bytes、permanent store；
- 保持现有 all-reduce；
- 正确性通过后与 equal TP 做 cold/warm benchmark。

该阶段提供后续所有复杂方案必须击败的低复杂度基线。

### Phase 2：静态 resident EP 原型

- 保持 cold host experts 使用 weighted TP；
- 选择少量 `(layer, expert)`，在启动时组装完整 expert 到一个 owner GPU；
- owner 过滤并执行这些 routes，其他 rank 不计算其 TP shard；
- 继续使用 output all-reduce；
- 初期禁用 CUDA Graph，验证数学正确性和完整 I=2048 kernel 性能；
- 优先验证 GPU3 resident-only 角色。

这个阶段不要求任意 GPU 能读取整个 host expert pool，因此避免立即解决共享 host source。

### Phase 3：Distributed expert directory

- 将 placement 从 layer 粒度下沉到 `(layer, expert)`；
- 建立 owner、slot、in-flight 和 replica 状态；
- 支持 rank-specific cache/permanent budgets；
- 定义跨 rank 原子提交和失败恢复；
- 增加可观测性和离线 trace replay。

### Phase 4：动态 EP cold fills

- 选择 shared host bank、transfer broker 或 static host ownership 方案；
- 支持 full expert DMA 到运行时选择的 GPU；
- 实现多 miss 的最小最大完成时间调度；
- 加入 cache value 和 eviction penalty；
- 对比 weighted TP，找出 EP 有利的 miss count/batch/cache 区域。

### Phase 5：同层 copy/compute pipeline

- per-GPU copy/compute queues；
- slot-level CUDA events；
- copy-next/compute-current；
- 与 shared expert 和 CPU branch 重叠；
- 固定容量 worklists 和 device valid counts；
- 逐步恢复 CUDA Graph capture。

### Phase 6：统一 TP/EP/CPU policy

- 对每个 miss set 选择 weighted TP、dynamic EP 或 CPU；
- 使用全机 host-bandwidth model；
- 支持 resident EP hit 与 cold TP 同层并存；
- 加入策略迟滞，避免 ownership/cache 抖动；
- 在真实 agent traces 上调参，而不是只依赖随机 token benchmark。

### Phase 7：NVLink transport（可选）

- 检测实际 NVLink pairing 和 P2P bandwidth；
- 增加 H2D direct 与 H2D+P2P relay 两种 transport；
- 将 staging VRAM 纳入 placement budget；
- 验证 relay 是否真正绕开 GPU3 x4，而不是只改善 NCCL。

## 10. 可观测性要求

后续实现必须至少提供以下 per-rank/per-layer 指标：

- routed unique experts、hits、misses；
- permanent/dynamic/CPU/TP/EP route counts；
- H2D bytes、copy duration、effective GB/s；
- direct H2D 与 relay P2P bytes；
- copy queue depth 和 compute queue depth；
- full expert 与各 TP width 的 kernel duration；
- CPU branch bytes/time；
- host bandwidth aggregate estimate；
- all-reduce duration；
- layer critical rank；
- scheduler decision 和预测/实际完成时间误差；
- cache eviction、replication、migration 和 in-flight reuse；
- TTFT、TPOT、request throughput、p50/p95/p99。

没有这些指标时，无法判断优化是减少了 x4 stall，还是仅把瓶颈转移到 GPU2 compute、
host DRAM、cache miss 或 collective。

## 11. 验证矩阵

### 11.1 正确性

- TP1 reference 与 equal TP4；
- equal TP4 与 weighted TP4；
- weighted TP 与 static resident EP；
- mixed resident-EP/cold-TP；
- mixed GPU/CPU partial output；
- prefill/decode、hash/scored router；
- cold cache、warm cache、eviction、runtime rebuild；
- 容许不同归约顺序产生的小范围浮点误差，但不得改变 expert routes 或出现系统性漂移。

### 11.2 性能场景

| 场景 | 目的 |
|---|---|
| 1 token decode，全部 miss | 测单 miss/少量 miss 关键路径 |
| 1 token decode，部分 hit | 测 resident compute 与 copy overlap |
| warm agent trace | 测真实 routing locality |
| 4K/8K/12K/16K prefill | 测 full-layer overlap 与 TTFT |
| concurrency 1 | 优化交互延迟 |
| concurrency 4/8 | 检查 EP 队列填充和吞吐潜力 |
| cache 256/512 | 分离 placement 与 miss-rate 收益 |
| graph off/on | 量化动态图调度成本 |
| GPU3 cold-fill on/off | 验证 resident-only 策略 |

### 11.3 验收原则

- 新方案必须保持服务正确性和 OpenAI API 行为；
- 不得通过系统 swap 换取模型可启动；
- host RAM、pin quota 和 VRAM 都必须在 planner 声明预算内；
- weighted TP 必须优于或至少不劣于 equal TP 后，才进入 EP 阶段；
- hybrid 必须优于最佳 weighted-TP baseline，而不是仅优于旧 equal TP；
- decode 改善不能以不可接受的 TTFT 或 context capacity 损失为代价；
- 所有结论同时报告 cold/warm cache 和 overlap 配置。

## 12. 风险

### 12.1 调度开销抵消收益

Decode 每层只有 6 个 routes，43 层重复执行。任何 host synchronization、Python 调度或
小消息 collective 都可能抵消几毫秒级 H2D 改善。最终控制流需要尽量 device-resident、
固定形状并兼容 CUDA Graph。

### 12.2 EP 任务粒度过粗

完整 expert 是约 12.75 MiB 的原子传输。少量 misses 时无法精确按带宽比例平衡，可能
出现 owner collision。必须保留 weighted TP fallback。

### 12.3 GPU2 compute 成为新瓶颈

`2:2:3:1` 会让 GPU2 计算 768-wide shard。若 SM75 FP4 路径更偏 compute-bound，
GPU2 可能替代 GPU3 成为 critical rank。需要以 kernel profile 而不是 PCIe 比例决定。

### 12.4 Host RAM 被重复映射或 pin

Dynamic full-expert EP 若错误地让每个 rank 各保存一份完整 host pool，将需要约 548 GiB
expert backing，完全不可行。共享 source 的物理页、pin quota 和进程生命周期必须在
架构阶段验证。

### 12.5 Aggregate host bandwidth

同时启用多 GPU DMA 和 CPU experts 可能使 DRAM 饱和。独立测得的各 GPU H2D 带宽不能
直接相加作为 planner capacity。

### 12.6 Cache locality 被 ownership 破坏

静态 owner 可能与 workload 热点不匹配；频繁迁移 owner 又会增加传输。需要 replica、
迟滞或历史价值模型，并以真实 agent trace 验证。

### 12.7 CUDA Graph 兼容性

动态 GPU owner、变长 work queue、跨设备 events 和 CPU branch 都会增加 graph capture
复杂度。原型阶段可使用 `--cuda-graph-max-bs 0`，但最终性能评估必须包含 graph-enabled
路径。

### 12.8 上游可接受性

SM75、特定异构 PCIe 比例和 multi-GPU dynamic EP 都可能超出上游当前目标。应保持：

- SM75 兼容补丁独立；
- 通用资源 placement 独立；
- weighted TP/EP 作为独立实验分支；
- topology policy 通过 capability/cost interface 注入，不将 `2:2:3:1` 写死在模型代码。

## 13. 当前建议

按风险和可验证性排序：

1. 恢复 prefill overlap，建立正确 TTFT baseline；
2. A/B 测试 cache 256/512，在显存预算内量化 miss-rate 收益；
3. 实现 MoE-only `2:2:3:1` weighted TP；
4. 建立 per-layer/per-rank copy 与 compute telemetry；
5. 实现少量 static resident EP，验证 full-expert kernel 和 replicated-input all-reduce；
6. 让 GPU3 试验 resident-only role；
7. 只有在 resident EP 确认有收益后，再实现 shared host source 和 dynamic EP；
8. 最终以 resident EP + cold dynamic TP/EP/CPU 作为长期目标；
9. NVLink 到位后，将 relay 作为新的 transport 纳入同一 scheduler。

近期最重要的决策不是“选 TP 还是 EP”，而是建立一种可表达以下事实的资源模型：

> 同一个专家在不同 residency、miss count、batch、链路状态和 cache 状态下，最优执行
> 方式可能不同。

只要 source-of-truth、容量预算、数学等价和同步边界保持清晰，FreeToken 的动态资源
管理思想就可以从单 GPU 的 CPU/GPU 二选一，扩展成多 GPU 上的 TP/EP/CPU 联合调度。

## 14. 开放问题

1. SM75 full-intermediate expert kernel 与 256/512/768 shard 的真实时间曲线如何？
2. 四张 GPU 同时 H2D 时，每条链路和 aggregate DRAM bandwidth 分别是多少？
3. 当前 cache 256 在真实 agent trace 上每层、每 rank 的 miss distribution 是什么？
4. GPU3 多保留 permanent experts 与减少 dynamic cache，哪种收益更高？
5. full expert host source 应使用共享映射、transfer broker，还是 static host ownership？
6. replicated-input EP 的 route filtering 能否保持固定形状并进入 CUDA Graph？
7. resident EP 与 cold weighted TP 同层混合时，最少需要几次 collective？
8. 多 GPU `q*` 应如何对共享 DRAM bandwidth 和 CPU cores 联合建模？
9. 是否值得对前三个 hash layers 实现精确提前预取？
10. NVLink 实际连接方式能否为 GPU3 提供可用 relay，还是只能连接部分 GPU pair？
11. 对 concurrency 1 和 concurrency >1，是否应使用不同 policy？
12. 如何定义 topology-neutral CLI，使上游不需要知道具体 GPU 编号和 lane 数？

## 15. 参考资料与当前实现入口

- [FreeToken paper: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)
- [现有 MoE 资源 placement 计划](moe-resource-placement-plan.md)
- [DeepSeek-V4 多 GPU 与资源布局说明](models.md#deepseek-v4-on-multiple-gpus)
- [`DSV4OffloadMoELayer` 与当前 MoE all-reduce](../python/freetoken/models/deepseek_v4/moe.py)
- [DSV4 routed-expert bank specs 与 TP shard loader](../python/freetoken/models/deepseek_v4/weight.py)
- [当前 distributed communicator 接口](../python/freetoken/distributed/impl.py)
- [Expert bank provider](../python/freetoken/moe/expert_banks.py)
- [Host bank 与 residency 实现](../python/freetoken/moe/host_banks.py)
- [Dynamic offload cache](../python/freetoken/moe/offload_cache.py)
