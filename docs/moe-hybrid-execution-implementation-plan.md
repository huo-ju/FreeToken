# MoE 异构混合执行分阶段实施计划

> 状态：实施中（P0 基线与 P1–P2 公共契约首批代码已落地）
> 记录日期：2026-08-24
> 目标分支：`work/moe-heterogeneous-execution`
> 首个目标平台：4 × NVIDIA TITAN RTX（SM75）单机，DeepSeek-V4-Flash-0731
> 相关文档：
> [异构 PCIe 环境下的 MoE 并行与专家执行设计](moe-heterogeneous-pcie-parallelism-design.md)、
> [MoE 多层资源管理与弹性 Placement 改造计划](moe-resource-placement-plan.md)、
> [FreeToken 论文](https://arxiv.org/abs/2608.16157)

## 1. 执行摘要

最终目标不是提供彼此独立的 TP、EP 和 CPU 三种运行模式，而是建立一个能够根据
硬件拓扑、资源容量、专家 residency、cache 状态、route 分布和运行时负载，为每组
routes 选择最合适执行方式的统一异构调度框架。

TP、EP 和 CPU 仍然必须分别可执行、可验证、可测量，但它们在新架构中是：

- 可组合的 execution actions；
- 自动策略的成本校准样本；
- 正确性与性能回归的 forced-policy baselines；
- 资源不足、策略失准或高级能力不可用时的安全 fallback。

实施路线采用“统一框架优先”的纵向切片，而不是“完整 TP -> 完整 EP -> 最后混合”
的瀑布方式：

1. 先冻结基线、补齐可观测性并定义统一 execution-plan contract；
2. 建立全节点、rank-aware 的资源和 topology planner；
3. 将 weighted TP 作为通用 cold-expert GPU action 接入；
4. 将现有 CPU executor 接入同一 plan，形成第一个 TP shard 逐 rank 选择 GPU/CPU
   的真实混合闭环；
5. 增加最小 resident EP action，并验证同层 TP + EP + CPU 数学等价；
6. 用 forced-policy 数据建立静态成本模型，再启用自动选择；
7. 最后才增加动态 ownership、cold EP fill、copy/compute pipeline 和 P2P relay。

每个阶段都有独立退出条件。没有达到退出条件时，不进入依赖它的复杂阶段。

## 2. 目标、边界与基本决策

### 2.1 目标

1. 适配 GPU 数量、显存容量、PCIe 带宽、CPU 算力、RAM 和 pin quota 不同的单机
   系统，不在模型代码中硬编码某一机器的 GPU 编号或 lane 数。
2. 在同一 MoE 层内，先为每条 logical route 选择 sharded 或 full-owner 表示；对
   sharded route 再逐 rank 选择由 GPU 还是 CPU 计算 local shard，并保持与原模型
   完全相同的数值语义。
3. 将 startup authoritative placement 与 runtime execution decision 分离：前者保证
   容量和 source-of-truth，后者只在合法候选 action 中选择。
4. 每种 action 都能被强制执行，以支持正确性验证、性能标定、故障隔离和回退。
5. 所有生产 decode 路径最终保持 fixed-capacity buffers、device valid counts 和
   CUDA Graph capture 能力。
6. 调度决策必须可解释：记录候选动作、预测成本、选择原因和实际完成时间。
7. 以最佳 forced policy，而不是当前 equal TP，作为自动混合策略的性能基线。

### 2.2 第一阶段非目标

- 不为 TP、EP 和 CPU 各建一套独立 cache、directory 或 scheduler。
- 不在每个 decode step 动态改变 expert tensor 的 TP shard width。
- 不立即实现任意 expert 的运行时 GPU ownership 迁移或复制。
- 不立即实现 cold full-expert EP fill、共享 host expert pool 或跨 rank transfer broker。
- 不立即支持 NVLink/P2P relay；它们属于统一 transport interface 的后续实现。
- 不一次性支持全部模型、量化格式和 FTW physical layout。
- 不以近似路由、expert skip、expert substitution 或改变精度换取性能。
- 不在没有 telemetry 的情况下实现复杂 copy/compute pipeline。

### 2.3 启动时静态、运行时动态的边界

第一版采用以下边界：

- routed expert 的 weighted-TP layout 在模型加载前确定；
- 每个 rank 的 host bank、dynamic slot 和 TP permanent store 使用该 rank 的固定
  local width；
- resident EP experts 使用独立的 full-expert representation；
- 运行时可按 `(layer, expert, route-set)` 在已有的 sharded/full representations 间
  选择，并逐 rank 决定 sharded work item 的 GPU/CPU target；
- 只有在 safe point 才能调整 cache/KV 容量、resident set 或 plan epoch；
- 任意需要新权重表示或 ownership 迁移的决策都不进入 per-token hot path。

该边界保留了主要调度能力，同时避免为动态 TP layout 保存多套 host 权重或在每个
token 上重新分片。

### 2.4 与现有 `hybrid` backend 的关系

当前 `--moe-backend hybrid` 表示“GPU cache hit/有限 fill + CPU overflow”，并不是本文
所说的多 rank TP/EP/CPU 统一混合执行。

迁移期间保留现有 CLI 语义，并把它映射为统一 execution plan 中的一种兼容 policy。
新框架稳定前不复用或改变 `hybrid` 这个用户可见名称，避免静默改变现有行为。

## 3. 统一模型与不可破坏的约束

### 3.1 三层计划

新架构明确区分三类状态：

1. `ResourcePlacementPlan`
   - 启动阶段生成；
   - 决定权威副本、rank-local tensor layout、permanent allocation、host residency、
     dynamic cache、KV 和 activation/graph headroom；
   - 容量验证通过后才允许分配和加载。
2. `ExpertExecutionPlan`
   - 决定当前 layer/step 的 route representation、shard target、transport、worklist
     和 aggregation；
   - 只引用 placement 已声明可用的表示和资源；
   - forced policy 和 auto policy 生成相同格式的 plan。
3. `RuntimeExecutionState`
   - 保存 cache mapping、LRU、in-flight copy、队列状态、带宽观测和 plan epoch；
   - hot-path 更新必须 device-side 或通过 graph-safe host node 完成；
   - host 侧只能在定义好的 safe point 读取汇总和提交新 epoch。

placement planner 不负责逐 token 调度，execution scheduler 也不能创建未计入容量的
权重表示。

### 3.2 Action 与 transport 正交

不要用单个 `TP/EP/CPU` 枚举表达全部状态。至少拆分以下维度：

| 维度 | 第一版候选 |
|---|---|
| authoritative source | `GPU_PERMANENT`、`HOST_PINNED`、`HOST_LOCKED` |
| route representation | `SHARDED`、`FULL_OWNER` |
| weight representation | `TP_SHARD`、`FULL_EXPERT` |
| compute work item | `GPU_SHARD`、`CPU_SHARD`、`GPU_FULL_OWNER` |
| weight transport | `NONE`、`LOCAL_H2D`；后续增加 `P2P_RELAY` |
| residency | permanent、dynamic hit、cold miss |
| aggregation | local sum + 一次 TP-group `all_reduce` |

例如：

- dynamic cache hit 可以执行 `GPU_SHARD + NONE`；
- cold host shard 可以执行 `GPU_SHARD + LOCAL_H2D`；
- resident full expert 可以执行 `GPU_FULL_OWNER + NONE`；
- rank-local host shard 可以执行 `CPU_SHARD + NONE`；expert weight 不移动，只传
  activation/result；
- 后续的 cold EP 才会出现 `FULL_EXPERT + LOCAL_H2D/P2P_RELAY`。

调度必须使用两级决策粒度。对一条 logical route `j`：

```text
SHARDED:
  y[j] = sum_r y_shard[j, r]
  每个 rank r 的 shard work item 必须选择 GPU_SHARD 或 CPU_SHARD，且只执行一次

FULL_OWNER:
  y[j] = y_full[j, owner]
  只有一个 owner 执行，所有 TP shard work items 都必须跳过该 route
```

因此本文的“TP + CPU 混合”不是把 logical routes 简单分成 TP 集合和 CPU 集合，而是
对 `SHARDED` route 的每个 rank-local shard 选择 GPU 或 CPU。不同 rank 可以因为
cache/link 状态不同而选择不同 compute target，但它们共同完成同一条 sharded route。
一条 route 不能同时使用 `FULL_OWNER` 和任意 TP shard，否则会重复计算。

### 3.3 核心不变量

所有阶段必须持续验证以下约束：

1. **Representation exactly once**：每个有效 `(token, topk-position)` 必须且只能
   选择 `SHARDED` 或 `FULL_OWNER` 中的一种表示。
2. **Shard work item exactly once**：对 `SHARDED` route，每个所需 rank-local shard
   必须且只能由 `GPU_SHARD` 或 `CPU_SHARD` 之一计算。
3. **TP coverage**：`offset[0] == 0`，相邻 shard 无洞无重叠，所有 local widths 之和
   等于 routed expert intermediate size。
4. **EP ownership**：执行 full expert 的 route 只有一个 owner rank，所有 rank 对该
   route 的 TP-shard partial 都必须为零。
5. **Exact aggregation**：GPU/CPU shard partial、EP owner full output 和
   shared-expert TP partial 在 local accumulation 后只执行一次定义明确的 collective。
6. **Single authoritative version**：每个 logical expert 只有一个权威权重版本；它可以
   是一组完整覆盖的 distributed host shards。full GPU representation、dynamic slots
   和其他物理表示必须带相同 version/checksum，并明确标记为 replica 或新的
   authoritative placement。唯一版本不等于只能有一份物理副本。
7. **Fallback eligibility**：只有完整、可访问的 TP shard source 仍然存在时，scheduler
   才能为该 expert 选择 sharded GPU/CPU fallback；如果 planner 为节省 host RAM 释放
   了这些 shards，则必须原子撤销该 action eligibility。
8. **Capacity before allocation**：所有 permanent、replica、host、pin、dynamic、KV、
   activation 和 graph reserve 在加载前完成全局证明。
9. **Plan agreement**：所有 TP ranks 使用相同 `plan_epoch` 和一致的 route
   representation/owner；每个 rank 的 GPU/CPU target 是全局 plan matrix 的本地切片。
   shard target 可以依据 rank-local graph-safe state 独立生成，不要求每步 all-gather；
   共同字段不一致必须 fail-fast，不能继续 collective。
10. **Graph safety**：生产 decode 不因真实 routing 执行 per-step Python 分支、动态形状
   allocation 或 device-to-host 同步。

### 3.4 建议的核心 contract

以下只是语义草图，具体 Python/C++ 类型在实现评审时确定：

```text
RoutedTpLayout
  widths[rank]
  offsets[rank]
  alignment
  total_intermediate_size

ExpertRepresentation
  layer_id, expert_id
  kind: TP_SHARD | FULL_EXPERT
  owners/ranks
  source tier
  bytes and version

ExpertExecutionPlan
  plan_epoch
  layer_id
  route_representation: SHARDED | FULL_OWNER
  full_owner_by_route
  shard_target[rank, route]: GPU_SHARD | CPU_SHARD
    # runtime rank only needs its local row; no per-step all-gather
  local_gpu_shard_worklist / valid_count
  local_cpu_shard_worklist / valid_count
  local_full_owner_worklist / valid_count
  copy descriptors and event dependencies
  aggregation contract

TopologyCostProfile
  rank identity by PCI BDF, NUMA and link properties
  H2D curves by payload/layout/concurrency
  GPU expert curves by local width/token count
  CPU curves by expert/token count and concurrent host traffic
  collective curves by message size/concurrency
```

`ExpertExecutionPlan` 的运行时表示必须是预分配 tensor/buffer，而不是每步创建 Python
对象。Python dataclass 可以用于配置、验证和测试，但不是最终 hot-path ABI。

## 4. 优先级与阶段依赖

| 优先级 | 阶段 | 主要结果 | 进入下一阶段的硬门槛 |
|---:|---|---|---|
| P0 | 基线与可观测性 | 能解释每层 critical path | 数据稳定且无明显测量扰动 |
| P1 | 统一 plan contract 与 forced policy | GPU/CPU shard 经同一接口执行 | reference 正确、原路径无显著回退 |
| P2 | 全节点 topology/resource planner | rank-aware、全局容量证明 | 计划一致、合成拓扑测试通过 |
| P3 | Weighted TP action | 通用 cold GPU fallback | 正确性通过且相对 equal TP 有实测价值 |
| P4 | CPU shard action 与首个混合闭环 | TP shards 逐 rank 选择 GPU/CPU | work-item exactly-once、overlap 和 graph 通过 |
| P5 | Resident EP 与三路静态混合 | TP + EP + CPU 同层执行 | 击败适用场景下最佳 forced baseline |
| P6 | 静态成本模型与 replay policy | 可解释的自动选择 | 预测误差受控且不差于 forced oracle |
| P7 | Graph-safe 自适应 scheduler | 运行时状态感知调度 | 长 trace 稳定、p95/p99 无不可控回退 |
| P8 | 高级 transport/迁移/pipeline | 扩展硬件与并发适应范围 | 每项单独由测量证明收益 |

P0-P2 是公共基础，不能被任何性能优化绕过。P3-P5 交付执行能力，P6-P7 才交付
自动调度能力。P8 中各项互不构成默认承诺。

## 5. P0：冻结基线与建立可观测性

### 5.1 目的

建立能够回答“时间花在哪里、哪个 rank 决定 layer latency、优化是否移动了瓶颈”
的证据闭环。该阶段不改变执行语义。

### 5.2 实施内容

1. 增加统一 TP4 server benchmark harness，固定并输出：
   - commit、完整 CLI、模型和 checkpoint 格式；
   - prompt/input hash、输入/输出长度、并发度和随机种子；
   - cache 初始状态、prefill overlap、CUDA Graph 状态；
   - GPU PCI BDF、NUMA、协商后的 PCIe generation/width；
   - host available/locked/pinned memory 和 swap delta。
2. 在不进行 per-layer host sync 的前提下记录：
   - active/unique/hit/miss/fetched experts；
   - 每 rank 每层实际 H2D bytes 和 copy duration；
   - routed expert compute、CPU compute、shared expert 和 all-reduce duration；
   - layer critical rank、layer wall time；
   - cache eviction、permanent hit 和 prefill buffer 行为。
3. 复用现有 device-side decode miss counters；新增 timing 时使用 CUDA events、NVTX
   或低频汇总，确保 metrics disabled 时不改变 graph 或热路径。
4. 扩展 copy/kernel microbench：
   - local intermediate width：256、512、768、2048；
   - miss count：1 到 top-k，以及 representative multi-token batches；
   - 单卡与四卡并发 H2D；
   - copy only、compute only、copy + compute contention；
   - equal payload 与 `2:2:3:1` payload。
5. 固定 prefill/cache 矩阵：
   - cache 256 / overlap off；
   - cache 512 / overlap off；
   - cache 512 / overlap on；
   - cache 大于 512 / overlap on。
6. 分别报告 cold start、cold decode、warm cache、prefill 后首轮 decode 和稳定 agent
   trace，不用随机 token microbench 替代真实 trace。

### 5.3 交付物

- 可重复运行的多 rank benchmark 入口；
- topology-aware profile 文件格式，profile key 不再只使用 GPU 型号；
- per-layer/per-rank JSON 或 JSONL trace；
- 一份 equal TP 基线报告和 cache/overlap 结论；
- 预测 copy-tail 与实测 layer critical time 的初始误差报告。

### 5.4 退出条件

- 相同配置至少三次运行的稳态 TPOT/throughput 波动在约定范围内；
- 能识别每层 critical rank，并区分 copy、compute、CPU 和 collective；
- metrics disabled 的性能与改造前无显著回退；
- 能证明 GPU3 x4 是否仍是主要可优化 critical path；
- 如果 GPU3 copy 并非主要瓶颈，暂停 weighted TP 性能工作，先修正假设。

## 6. P1：统一 execution plan 与 forced-policy 框架

### 6.1 目的

建立所有执行 action 共用的 route representation、shard worklist、local accumulation
和 aggregation contract。先用现有 equal TP 和 CPU-shard 路径证明接口，不立即
增加 EP。

### 6.2 实施内容

1. 定义 topology-neutral `RoutedTpLayout`，equal TP 通过它表达，而不再依赖到处
   调用 `div_even(intermediate_size, tp_size)`。
2. 定义 execution action、representation 和 plan epoch 的内部类型及验证器。
3. 为 decode 预分配每种 action 的固定容量 worklist、route mask、valid count 和
   output accumulation buffer。
4. 将当前 equal TP GPU path 适配为 `GPU_SHARD` action；最初允许 adapter 直接调用现有
   cache/GEMM 实现，避免一次重写全部热路径。
5. 将纯 CPU decode 适配为 `CPU_SHARD` action；每个 rank 的 CPU executor 继续读取
   自己的 local TP shard，并保留现有 graph-safe host-node executor。
6. 增加 forced policies：
   - `force_equal_tp`；
   - `force_cpu`；
   - 保留当前 `offload/cpu/hybrid` CLI 到兼容 policy 的映射。
7. 将 local accumulation 与最终 collective 放到明确的单一边界；DSV4 routed partial
   与 shared-expert partial 继续只执行一次 all-reduce。
8. 增加 debug-only plan dump 和 plan checksum；所有 rank 在 collective 前验证 epoch。

### 6.3 测试

- equal layout 的 width/offset 和旧实现完全一致；
- forced equal TP 对齐现有 TP4 输出；
- forced CPU 对齐 TP1/reference；
- route mask 为空、全部 routes、单 expert、多 token 和重复 expert route；
- graph capture/replay 使用不同实际 routing，valid count 正确更新；
- plan epoch 不一致、非法 representation 和 route 重叠时 fail-fast；
- compatibility CLI 的行为与改造前一致。

### 6.4 退出条件

- equal TP 与 CPU 两种 forced policy 都通过 reference；
- 关闭新调度能力时，现有用户行为不变；
- equal TP steady-state 性能没有超过约定阈值的回退；
- hot path 没有新增 per-step Python decision 或 device-to-host sync。

## 7. P2：全节点 topology 与资源 planner

### 7.1 目的

把当前每 rank 独立、平均拆分预算的 placement 改造成一次全节点联合规划，为
weighted TP 和 rank-specific residency 提供容量保证。

### 7.2 实施内容

1. 每个 rank 上报：
   - PCI BDF、NUMA、SM、VRAM、当前/最大 PCIe generation 和 width；
   - baseline free VRAM、fixed weights、graph/activation reserve；
   - 每种 local width 的 expert bytes 和 kernel profile；
   - 可访问 host source、pin 能力和 checkpoint capability。
2. rank 0 或独立纯函数 planner 联合求解，再广播不可变 plan；其他 rank 独立复算
   checksum 和容量约束。
3. 使用全局约束：
   - `sum(retained_host_bytes[rank]) <= host_budget`；
   - `sum(pinned_host_bytes[rank]) <= pin_budget`；
   - 每 rank 分别满足 VRAM 和 activation/graph headroom；
   - KV 使用所有 rank 中的实际可分配下界，不使用最大 free memory；
   - intentional asymmetry 由计划验证，不再用固定 2 GiB 差异直接拒绝。
4. 同时选择：
   - `RoutedTpLayout`；
   - per-rank permanent layer/expert budget；
   - per-rank dynamic cache bytes/slots；
   - full-expert replicas 及其额外 VRAM、source-retention 和 fallback eligibility；
   - host pinned/locked placement；
   - KV pages 和 prefill overlap 可行性。
5. layout 至少验证：
   - widths 总和等于完整 intermediate size；
   - 每个 width 满足对应 kernel tile/alignment；
   - zero-width rank 第一版不允许；
   - shared expert 保持现有 equal TP，不受 routed layout 影响。
6. 增加 synthetic topology 输入，使 planner 测试不依赖当前物理机器。
7. 第一版 weighted layout 只支持原始 DSV4 FP4 safetensors；FTW 或未知 layout
   明确拒绝，不静默加载错误分片。

### 7.3 初始搜索种子

当前机器可将以下方案作为搜索和容量测试种子，但不能硬编码成生产策略：

```text
routed TP width:      [512, 512, 768, 256]
width units / 256:    [2,   2,   3,   1]
permanent layer seed: [16,  16,  10,  32]
```

该组合的意义是展示 weighted TP 必须和 rank-specific residency 联合求解；最终选择
必须来自实测 bandwidth/compute curve 和容量约束。

### 7.4 测试

- 相同 GPU/带宽得到 equal layout；
- x16/x8/x8/x4、不同 VRAM、不同 host/pin budget 的合成案例；
- global budget 可行但平均 per-rank budget 不可行的案例；
- layout alignment、host OOM、rank VRAM OOM、KV reserve 不足；
- planner 决定确定性、广播 checksum、错误 rank plan 拒绝；
- cache 256/512/大于 512 与 overlap floor 的容量组合。

### 7.5 退出条件

- allocation 前能输出完整 per-rank 资源证明；
- 所有 worker 对 plan byte-for-byte 一致；
- 当前 equal TP 配置仍可得到与旧 planner 等价的可行方案；
- weighted candidate 在真实模型加载前被证明满足 VRAM、host、pin 和 KV 预算；
- 任何不支持的 checkpoint/layout 都在分配前 fail-fast。

## 8. P3：Weighted TP action

### 8.1 目的

提供适应异构 PCIe/compute 能力的通用 GPU cold-expert action，并成为后续所有混合
策略的安全 fallback 和性能基线。

### 8.2 实施内容

1. 按 `RoutedTpLayout` 泛化 DSV4 routed-expert bank specs、loader slice、offset 和
   checkpoint streaming。
2. 让 `OffloadMoeCache`、dynamic slot bytes、prefill buffers、copy descriptors 和
   TP permanent store 使用 rank-local specs。
3. 保持 routed kernel 从 bank shape 推导 local intermediate width；为 256、512、768
   增加显式 alignment/assertion 和针对 SM75 的 profile。
4. shared expert 继续 equal TP；routed weighted partial 与 shared partial 在本地相加后
   继续使用一次 all-reduce。
5. 增加 `force_weighted_tp`，同时保留 `force_equal_tp` 作为同框架基线。
6. 第一版 layout 只在 startup 固定，不在请求期间改变。
7. 加载日志输出每 rank width、offset、expert bytes、host bytes、permanent bytes、
   dynamic cache bytes 和剩余 KV/headroom。

### 8.3 正确性测试

- TP1 full expert reference；
- equal TP4 与现有结果；
- `512,512,768,256` partial outputs 求和对齐 unsharded reference；
- offset 首尾、无洞、无重叠以及非法 tile；
- prefill/decode、hash router/learned router、cold/warm cache；
- permanent layer、host-backed layer、cache eviction 和 cache rebuild；
- rank-specific permanent sets 不同但同层结果一致；
- 多 worker 真实 checkpoint 小样本集成测试。

### 8.4 性能门槛

在相同 context、KV、cache bytes、prefill 配置和 concurrency 下比较最佳 equal TP：

- 不允许通过减少 KV、缩短上下文或增加 swap 获得结果；
- 报告 copy-only、layer critical time、TTFT、TPOT、throughput 和 p95；
- 建议合入 auto 候选集的初始门槛为稳态 TPOT/throughput 改善至少 10%，且 TTFT/p95
  无超过 5% 的不可解释回退；
- 如果只改善 copy microbench、但端到端未改善，保留 forced experimental action，
  不进入默认 auto policy。

## 9. P4：CPU shard action 统一化与第一个混合闭环

### 9.1 目的

在不引入 EP 的情况下，先证明统一 plan 可以为每条 sharded route 的各 rank-local
shard 独立选择 GPU 或 CPU，并能重叠两条执行路径。这是最早可交付的真正混合纵向
切片。

### 9.2 实施内容

1. 将现有 `HybridMoeBackend` 的 local GPU hit/fill 与 local CPU overflow partition
   表达成统一 `ExpertExecutionPlan`，不再成为独立的特殊 decision tree。
2. 对每个 rank，以 raw expert ids 生成互斥的 local GPU/CPU shard masks：
   - `GPU_SHARD` 包含该 rank 的 cache hits 和本步选择 fill 的 misses；
   - `CPU_SHARD` 包含该 rank 的剩余 misses；
   - 对每条 `SHARDED` route，每个 rank 的两种 mask 并集为一个 local shard work
     item，交集为空；
   - 不要求不同 rank 选择相同 compute target，但 route representation 必须一致。
3. GPU 和 CPU 都使用当前 startup `RoutedTpLayout` 对应的 rank-local width/source；
   CPU executor 计算 local shard partial，而不是完整 expert output。
4. 保留 CPU submit -> GPU copy/GEMM -> CPU sync 的 overlap；serial 模式只作为 A/B
   诊断开关。
5. 将论文式 `q*` 从单 rank 常数扩展为全机候选策略：考虑四个 rank 同时 H2D 对
   host DRAM、CPU executor 和 PCIe 的竞争。
6. 第一版自动规则仍可使用静态 fetch cap/fraction，但必须通过同一 plan contract；
   `force_weighted_tp` 和 `force_cpu` 仍可随时绕过策略。

### 9.3 测试

- 每个 rank 上 GPU-only、CPU-only 和从 0 到全部 local shard work items 的 split 点；
- representation exactly-once、shard work-item exactly-once、权重归一化和 partial
  merge；
- CPU capability/health 在生成 plan 前不可用时选择 `GPU_SHARD`；运行中的 CPU
  timeout 必须 fail loud，并在 safe point 禁用后续 CPU plans，不能用 stale output
  静默续跑；
- graph capture/replay 中 routing 和 split 改变；
- overlap on/off 数值一致；
- 四 rank 不同 cache hit/miss 状态下允许不同 local target，但仍使用一致的
  `SHARDED` representation；
- shared expert 与 routed GPU/CPU shard partial 最终只执行一次 collective。

### 9.4 退出条件

- 同层各 rank 的 GPU_SHARD + CPU_SHARD 输出求和后对齐 reference；
- CUDA Graph 可捕获并可用不同 routing 重放；
- forced TP、forced CPU、serial mixed 和 overlapped mixed 都可独立测量；
- 至少存在一个实测区域使 mixed 不差于两个 forced baselines，否则不启用默认自动
  split，但保留 action 和测量能力。

## 10. P5：Resident EP action 与三路静态混合

### 10.1 目的

增加最小 full-expert owner execution，使稳定热点专家可以避免运行期 H2D，并验证
同层 `resident EP + cold weighted TP shards（GPU/CPU）` 的完整数学和同步模型。

### 10.2 实施内容

1. 增加独立 `FullExpertPermanentStore`；不能把 `I=2048` full expert 混入现有
   rank-local TP store 的统一 bank shape。
2. 第一版 full-expert assembly 采用明确、可审计的启动路径：
   - 推荐 owner rank 从原始 safetensors 直接读取完整 expert；
   - 不依赖运行时共享 host source；
   - 不支持的 checkpoint 格式在加载前拒绝。
3. 第一版保留对应的 rank-local host shards，并把 full GPU representation 标记为
   带 version/checksum 的 executable replica；这样同一 expert 仍可回退到 sharded
   GPU/CPU。当前 host banks 为 layer-granular，在没有安全的 per-expert release 机制
   前，不释放单个 resident expert 的 host rows。
4. startup planner 为每个 resident `(layer, expert)` 指定唯一 owner、额外 VRAM bytes、
   replica version 和 source-retention/fallback 规则。后续如果释放全部对应 host
   shards 并把 full GPU copy 提升为 authoritative placement，必须同时撤销该 expert
   的 sharded/CPU eligibility。
5. router 输出在 device-side 做两级分解：
   - route-level：`FULL_OWNER` 与 `SHARDED` 互斥且覆盖全部有效 routes；
   - `FULL_OWNER`：仅 owner 运行 full expert，所有 TP shard work items 关闭；
   - `SHARDED`：每个 rank 必须产生一个 local shard work item，再根据该 rank 的
     cache/link/CPU 状态选择 `GPU_SHARD` 或 `CPU_SHARD`。
6. 各 rank 将本地 GPU/CPU shard partial、自己拥有的 EP full output 和 shared expert
   partial 累加，沿用一次 TP-group all-reduce。
7. 先支持启动时静态 resident set，不支持运行时迁移；选择依据来自离线 trace 或
   人工 forced plan。
8. `force_resident_ep` 用于只路由到已 resident experts 的 microbench；
   `force_static_mixed` 用固定 plan 验证三路组合。

### 10.3 测试

- full expert kernel 对齐 TP1/reference；
- owner 为任意 rank，其他 rank 不重复计算该 route；
- 同一 token 的 top-k 同时包含 `FULL_OWNER` 和 `SHARDED` routes，并让不同 rank 的
  sharded work items 分别落到 GPU/CPU；
- full/sharded 某组为空、某 rank 的 GPU/CPU 某组为空、多个 owner 和多 token；
- EP full output 与 TP partial 在一次 all-reduce 中正确合并；
- full store 与 TP permanent/dynamic store 容量和生命周期隔离；
- checkpoint 加载失败、owner 不一致、重复 ownership 和 stale epoch fail-fast；
- graph-off correctness 通过后，再增加 graph capture/replay 测试。

### 10.4 Go/no-go 门槛

只有同时满足以下条件，resident EP 才进入 auto 候选集：

- 真实 agent trace 显示可利用的稳定 layer/expert 热点；
- full-expert kernel 和 collective 没有抵消避免 H2D 的收益；
- 在相同总 VRAM/KV 预算下击败最佳 weighted TP/CPU mixed baseline；
- resident set 对 workload 漂移不过度敏感，或有明确安全回退；
- graph-enabled 路径没有不可接受回退。

未达到门槛时，保留 static experimental action，不继续 dynamic EP。

## 11. P6：静态成本模型与离线 replay policy

### 11.1 目的

把 forced-policy 测量转成可解释的机器级 cost model，并在不做动态 ownership 的
前提下自动选择 route representation，以及每个 sharded work item 的 GPU/CPU target。

### 11.2 成本模型

至少建模：

```text
T_rank[r] = resource_schedule(
    T_copy[r], T_gpu_shard[r], T_cpu_shard[r], T_full_owner[r], queue/events[r]
)
T_layer(plan) = max_r(T_rank[r]) + T_collective + T_scheduler
```

实际模型必须允许 copy/compute/CPU 的已验证 overlap，不能简单相加，也不能假定
理论 PCIe 带宽。输入至少包括：

- 每 rank link/profile 和当前 queue depth；
- local TP width、hit/miss expert count 和 token/route count；
- resident EP owner、full-expert kernel time；
- CPU cores、expert/route count 和 aggregate host bandwidth；
- cache value、eviction penalty 和后续 reuse estimate；
- all-reduce cost；
- concurrency 和 prefill/decode 类型。

### 11.3 实施内容

1. profile 使用 PCI BDF、GPU/SM、NUMA、driver/runtime、tensor layout 和并发级别作为
   identity，不以 GPU 型号名称作为唯一 key。
2. 建立离线 trace replay：对同一 routing/cache trace 枚举合法 forced/mixed plans，
   得到 oracle 或近似下界。
3. 第一版 policy 使用确定性、可解释规则；每次输出候选成本、拒绝原因和最终选择。
4. 为 prediction error 建立按 action、rank、miss count 和 batch 分桶的报告。
5. 对 workload 漂移使用保守 guardrail：数据不足或预测置信度低时退回 weighted TP
   或已验证的 sharded GPU/CPU policy。

### 11.4 退出条件

- replay 对相同输入产生确定性 plan；
- cost prediction 能正确预测大多数 layer 的 critical rank/action；
- 预测误差达到预先约定范围，异常有可解释分桶；
- static auto 在代表性 traces 上不差于适用的最佳 forced baseline，或差距在明确的
  scheduler overhead 预算内；
- policy 不产生容量之外的表示、copy 或 residency。

## 12. P7：Graph-safe 自适应混合 scheduler

### 12.1 目的

让 scheduler 使用实时 cache、queue 和带宽状态调整 execution actions，同时保持
fixed-shape 和 CUDA Graph 能力。

### 12.2 实施内容

1. 将 action eligibility、route representation、shard target 和 worklist valid count
   放到 device-side；
   host 不读取每步 routing 后再决定。
2. 第一版自适应边界为：
   - `FULL_OWNER/SHARDED` 由所有 rank 共享的 resident-action table、routing 和
     `plan_epoch` 确定，保证无需逐步通信即可一致；
   - 每个 rank 的 `GPU_SHARD/CPU_SHARD` target 可以依据本地 cache、queue 和健康状态
     自适应；
   - 基于瞬时 owner/远端队列动态切换 `FULL_OWNER/SHARDED`，只有在存在低开销、
     graph-safe 的跨 rank 协调机制后才允许。
3. 建立固定容量的 distributed expert directory，第一版只描述已有 representation：
   - TP shard source/cache slot；
   - resident EP owner/full slot；
   - CPU availability；
   - version、in-flight 和 plan epoch。
4. 使用 slot-level events 和明确 stream dependency，禁止通过全设备 synchronize 保证
   正确性。
5. scheduler 只在合法 action 集合中选择；若实时信息缺失或队列异常，执行确定性
   fallback。
6. 增加 hysteresis、最小 residency lifetime 和切换成本，防止 route representation
   或 GPU/CPU shard target 在临界值附近抖动。
7. concurrency >1 时加入 request fairness、host bandwidth sharing 和 queue deadline；
   concurrency 1 与服务吞吐策略可以使用不同参数，但共用 cost interface。
8. safe point 更新 profile、policy 参数或 resident set 时使用新 epoch 原子提交；旧
   in-flight plan 完成前不得回收其资源。

### 12.3 测试

- graph capture 后使用不同 routing、miss count 和 action split 重放；
- directory version/epoch 切换与 in-flight copy；
- queue saturation、CPU slowdown、PCIe contention 和 profile 缺失；
- fallback 始终生成合法 plan；
- 长 agent trace 中 cache、directory 和 route accounting 无泄漏；
- concurrency 1、2、4 及 prefill/decode 混合；
- scheduler overhead、p50/p95/p99 和 action churn。

### 12.4 退出条件

- graph-enabled 与 graph-off 数值一致；
- scheduler hot path 无 per-step host sync 或动态 allocation；
- 自动策略在多种代表性硬件 profile/trace 上不差于安全 fallback；
- 长时间运行无 stale ownership、double execution、cache corruption 或 collective
  divergence；
- p95/p99 回退在预设 guardrail 内。

## 13. P8：条件性高级能力

以下能力互不绑定，每项都必须有独立测量、设计和 go/no-go。

### 13.1 Dynamic cold EP fill

仅在 resident EP 已证明收益、且 trace 显示静态 resident set 覆盖不足时考虑：

- 选择 shared host mapping、transfer broker 或 static host ownership；
- 支持 full expert 到动态 owner 的 H2D；
- 将 full representation 的额外 host/VRAM bytes 纳入 planner；
- directory 支持 ownership 原子提交、失败恢复和 in-flight reuse；
- 与 weighted TP 比较 miss count、batch、owner link 和 eviction 区域。

如果 full-expert cold transfer 通常不优于多链路 weighted TP，则不实现动态 EP。

### 13.2 Copy/compute pipeline

当前 fused UVA gather 是 GPU kernel，不应假设它等价于独立 DMA engine。只有 P0/P3
数据证明存在可重叠空间时才实施：

- 分离 copy-next 与 compute-current worklists；
- 测量 copy kernel 与 expert kernel 的 SM/带宽竞争；
- 使用 slot events 而不是 layer-wide synchronize；
- 保持固定容量和 graph capture；
- 与未 pipeline 的最佳 mixed baseline 比较，而非只看 copy latency。

### 13.3 P2P/NVLink relay

只有检测到实际 peer connectivity 和端到端收益时加入：

- direct H2D 与 H2D + P2P relay 使用统一 transport contract；
- staging VRAM、额外 copy、event 和 source-link contention 纳入 cost model；
- 不把 NVLink 存在等同于 GPU3 一定可被 relay；
- relay 对 NCCL 的影响和 expert transport 的影响分别测量。

### 13.4 Dynamic replication/migration

仅在稳定热点、static resident EP 和 directory 均验证后考虑：

- replica value 必须超过复制成本和被挤出的 KV/cache value；
- 迁移只在 safe point 提交；
- 每个 representation 有清晰 authoritative/replica 关系；
- workload 漂移时有回收和回退策略。

## 14. 全阶段验证矩阵

### 14.1 数值与分解正确性

| 维度 | 必测组合 |
|---|---|
| reference | TP1 full expert、equal TP4、weighted TP4 |
| action | forced GPU shards、forced CPU shards、forced resident full EP |
| mixture | shard 内 GPU+CPU、sharded+full EP、三者组合 |
| routing | hash layers、learned router、单 expert、多 expert、重复 expert、多 token |
| lifecycle | cold、warm、eviction、permanent、rebuild、epoch 切换 |
| request | prefill、首 token、steady decode、长 agent trace |
| graph | graph off、capture、不同 routing replay |

测试必须同时检查最终输出和 route accounting：只对齐输出不足以发现重复执行后被
偶然抵消、漏 route 或错误 owner。

### 14.2 容量与资源正确性

- rank-local VRAM peak 与 planner 预测；
- aggregate host RSS、locked/pinned bytes 和 swap delta；
- permanent、dynamic、full EP 和 KV allocation 物理隔离；
- cache rebuild 前后 plan representation 仍合法；
- cache 256/512/大于 512 的 prefill buffer 与 decode warm-state；
- intentional rank asymmetry 不导致 KV OOM 或错误 collective；
- 加载失败不会留下半提交 ownership。

### 14.3 性能矩阵

至少覆盖：

- input length：短 prompt、4K、8K、12K/16K；
- output length：短输出和 1K 级长 decode；
- concurrency：1、2、4；
- cache：cold、warm、真实 agent reuse；
- prefill overlap：off/on；
- policies：所有 forced policies、deterministic mixed、auto；
- 指标：TTFT、TPOT、request throughput、p50/p95/p99、每层 critical path、CPU/GPU
  utilization、H2D/host bandwidth、all-reduce 和 scheduler overhead。

任何性能结论都必须保持 context、KV、cache bytes、模型精度和输出语义一致，并报告
是否发生 swap、OOM retry 或 fallback。

## 15. CLI、配置与兼容策略

### 15.1 过渡原则

- 现有 `--moe-backend offload|cpu|hybrid` 行为保持兼容；
- 新能力先通过 experimental execution-policy 配置暴露；
- `auto` 只有在对应 action 达到阶段退出条件后才允许选择它；
- 未 profile 的硬件默认使用保守 fallback，而不是套用 TITAN RTX 数据；
- topology policy 不包含固定 `GPU3`、`x4` 或 `2:2:3:1` 判断。

### 15.2 建议的实验配置面

具体命名在 CLI review 时确定，语义建议如下：

```text
moe_execution_policy:
  compatibility | force_equal_tp | force_weighted_tp |
  force_cpu | force_resident_ep | force_static_mixed | auto

moe_tp_layout:
  auto | explicit startup-only layout

moe_resident_ep:
  disabled | explicit offline set | planner-selected set

moe_cost_profile:
  topology-aware profile path/version
```

debug/forced 参数不保证长期用户 API 稳定，但其语义和测试能力必须保留。

## 16. 建议的 PR 拆分与代码落点

每个 PR 必须可单独回滚，不同时引入新的权重表示、调度算法和 pipeline。

1. **PR-A：benchmark 与 telemetry**
   - 多 rank harness、topology identity、trace schema、copy/kernel profiles；
   - 不改变执行决策。
2. **PR-B：execution-plan contract**
   - layout/action/plan types、forced equal TP/CPU adapters、route accounting；
   - 保持现有 CLI 行为。
3. **PR-C：joint resource planner**
   - rank inputs gather、global constraints、plan broadcast/checksum；
   - equal TP 先通过。
4. **PR-D：weighted TP action**
   - rank-local specs/loader/cache/store/kernel tests；
   - forced policy 和真实模型 benchmark。
5. **PR-E：CPU shard action 与统一 sharded GPU/CPU mixed plan**
   - 迁移现有 hybrid split，保持 graph-safe overlap；
   - forced oracle 对比。
6. **PR-F：resident EP representation/action**
   - full store、owner load、full kernel、forced EP；
   - 暂不自动选择。
7. **PR-G：静态三路 mixed plan 与 trace replay**
   - sharded GPU/CPU + full EP correctness、离线 oracle、cost profile。
8. **PR-H：auto policy**
   - 确定性 cost scheduler、guardrail、fallback、prediction telemetry。
9. **PR-I：graph-safe runtime adaptation**
   - device worklists/directory、epoch update、concurrency。
10. **后续独立 PR**
    - cold EP、pipeline、P2P relay、replication/migration；每项单独立项。

### 16.1 主要代码职责

| 代码区域 | 计划职责 |
|---|---|
| `python/freetoken/engine/moe_placement.py` | 纯函数式全节点容量约束与 placement 结果 |
| `python/freetoken/engine/engine.py` | rank resource gather、plan broadcast、allocation sequencing 和 fail-fast |
| `python/freetoken/layers/moe.py` | execution-plan 消费、local worklists、partial accumulation 和统一 collective 边界 |
| `python/freetoken/moe/offload_cache.py` | rank-local shard cache、copy descriptors、directory state 和 metrics |
| `python/freetoken/moe/cpu_executor.py` | `CPU_SHARD` execution，不承担全局策略决策 |
| `python/freetoken/moe/permanent_store.py` | 现有 TP-shard permanent store；full EP 使用独立 store/type |
| `python/freetoken/models/deepseek_v4/weight.py` | `RoutedTpLayout` 驱动的 specs、offset 和 checkpoint slice |
| `python/freetoken/models/deepseek_v4/moe.py` | DSV4 router/shared-expert 接线，不承载 topology policy |
| `python/freetoken/moe/bench_profile.py` | topology-aware copy/compute/CPU/collective profile schema |
| `tests/moe/`、`benchmarks/` | forced-policy oracle、组合正确性、容量与端到端回归 |

具体类型可放入新的公共模块，但不能让 model-specific loader、cache 和 scheduler 分别
维护不一致的 layout 或 action 定义。

## 17. 决策门与停止条件

### 17.1 Weighted TP 决策门

- 如果 P0 显示最慢 PCIe rank 不在大多数 MoE 层 critical path，停止以拓扑比例为核心
  的 weighted TP 优化；保留通用 layout contract，但重新评估首要瓶颈。
- 如果 weighted TP 只改善 copy 而恶化 compute/collective，使端到端没有收益，则不
  进入 auto policy。

### 17.2 Resident EP 决策门

- 如果热点不稳定，或 full expert owner compute 成为新瓶颈，则只保留实验实现；
- 如果相同 VRAM 用于 dynamic TP cache 的收益更高，不扩大 resident EP。

### 17.3 Dynamic EP 决策门

- 如果 cold full-expert transfer 在目标 miss/batch 区域不优于 weighted TP，不实现
  shared host source 和动态 ownership；
- static resident EP 未证明收益前，禁止进入 dynamic EP。

### 17.4 Scheduler 决策门

- auto policy 在代表性 trace 上不能接近 forced oracle 时，继续使用静态 policy；
- graph-safe 实现显著损失性能时，先修正 worklist/directory，不以永久关闭 graph
  作为完成标准。

## 18. 首轮实施清单

按实际开始顺序，第一轮只执行以下工作：

1. 固化当前 TP4 基线配置和真实 agent trace 输入；
2. 扩展 benchmark，使其支持 TP4 server、per-rank trace 和 topology identity；
3. 补齐 copy、expert compute、CPU、all-reduce 和 layer critical-rank telemetry；
4. 完成 cache 256/512/大于 512 与 prefill overlap 的正确 A/B；
5. 定义 `RoutedTpLayout`、action、execution plan 和 route-accounting invariant；
6. 用 equal GPU shards 和 CPU-shard adapter 验证统一 plan，不改变用户可见默认
   行为；
7. 将 placement planner 改成全节点联合求解，并先让 equal TP 回归通过；
8. 只有上述退出条件满足后，才开始 weighted TP loader/cache/store 改造。

首轮不实现 resident EP、dynamic directory、cold EP、P2P relay 或 copy/compute
pipeline。这样可以最早验证统一架构是否成立，同时把高风险工作建立在可观测、可回退
的公共基础上。

## 19. 完成定义

本计划的最终完成不是“TP、EP 和 CPU 三条路径都存在”，而是同时满足：

1. 三种 action 共用同一个 placement、execution-plan、directory、telemetry 和
   aggregation contract；
2. 每种 action 都能 forced 执行并对齐 reference；
3. 同层任意合法 route representation 和 shard-target 组合都满足 exactly-once 和
   容量约束；
4. auto policy 根据实际硬件和运行状态选择 action，并可解释、可回退；
5. 自动策略在目标硬件矩阵和真实 traces 上不差于适用的 forced baselines；
6. decode hot path 保持 fixed-shape、无 per-step host sync，并支持 CUDA Graph；
7. 新硬件通过 topology/profile/capability 接口接入，无需修改模型特定调度逻辑。

达到这些条件后，TP、EP 和 CPU 才真正从独立机制转化为统一异构资源调度系统中的
可组合能力。

## 20. 实施记录

### 2026-08-24：P0–P2 第一批公共基础

已完成：

- TP server benchmark 支持 `--tp-size`、默认三次稳态重复，并在 JSONL 中记录 commit、
  完整 server CLI、prompt hash、CUDA Graph/prefill/cache 状态、每次测量、GPU PCI
  BDF/NUMA/协商链路以及 host memory/swap delta；
- copy microbench 增加 DSV4 DS-FP4 的 local width 256/512/768/2048 profile 和精确
  `--miss-counts` sweep；
- 增加 topology-aware cost-profile v1 schema；相同 GPU 名称但 PCI BDF、NUMA、链路、
  layout 或 concurrency 不同不会复用 profile；
- 经实施复核决定不新增 per-layer/per-rank telemetry schema；继续使用 FreeToken 现有
  `/v1/stats`、device-side miss/fetch counters 和 benchmark 结果，不让后续执行框架
  依赖新的 tracing 子系统；
- 增加 `RoutedTpLayout`、action/representation/transport/policy 类型、plan epoch、稳定
  checksum 和 exactly-once 验证；
- 增加 fixed-capacity local plan tensor ABI；不同 routing/valid count 的 replay 原地更新，
  不替换 buffer storage；
- `force_equal_tp` 与 `force_cpu` 通过 experimental execution-policy 配置映射到现有
  GPU/CPU shard kernel；默认 `offload|cpu|hybrid` 保持 compatibility 语义；
- DSV4 equal-TP routed bank specs 和 checkpoint slice 开始统一使用 `RoutedTpLayout`；
  shared expert 仍保持原 equal TP；
- 增加纯函数式全节点 planner：按实际 rank-local shard bytes 分配 node-shared host/pin
  budget，逐 rank 证明 VRAM/dynamic/KV floor，使用全 rank KV 下界，并生成可复算 checksum；
- 增加 equal/weighted layout、异构 PCIe、全局预算、未知 weighted checkpoint、epoch、
  route overlap、fixed-buffer replay 和 telemetry accounting 的 CPU-only 测试。

验证结果：使用 `/home/huoju/work/freetoken-triton-turing-work/.venv` 和本 worktree 的
`PYTHONPATH` 运行全仓测试，结果为 `1446 passed, 25 skipped`。

P0 实测基线：

- checkpoint：`/data/models/DeepSeek-V4-Flash-0731`，TP4，固定 16K KV，cache 768，
  overlap off，CUDA Graph on；输入来自 `dsv4-real-eval-postfix-20260823` 的
  `decode_64_r0`；
- 三次稳态中位数为 7.523 tok/s、132.918 ms/token、TTFT 2.717 s，吞吐极差
  0.004 tok/s；三次 greedy 输出 hash 一致；
- 运行期间 host available 最低约 8.83 GiB，swap 无增长；实际峰值链路为
  Gen3 x8/x8/x16/x4；结果位于
  `benchmarks/results/moe-hybrid-p0-20260824/tp4-cache768-overlap-off-kv16k.jsonl`；
- TP>1 的 runtime cache rebuild 当前不受支持，benchmark 已在模型启动前拒绝多值
  `--cache-sequence`，避免为不可执行的矩阵反复加载 checkpoint。

P1 forced-policy adapter 接线：

- engine 初始化时创建不可变 `ExecutionPolicyAdapter`，将 forced policy 映射为统一
  `GPU_SHARD`/`CPU_SHARD` reference plan，并校验实际 decode executor 与 residency；
- `force_cpu` 禁止 GPU permanent layers；完整 CPU shard placement 不可行时必须在启动
  阶段失败，不能静默形成 GPU/CPU 混合；`force_equal_tp` 同样拒绝 CPU layer override；
- compatibility policy 仍完全委托现有 backend decision tree，不增加 decode 热路径分支。

P2 精简 node planner 接线：

- 每个 rank 在 expert allocation 前仅上报本地 VRAM、fixed-weight bytes、local shard
  bytes、provider capabilities 和基础 PCI identity；不做 topology 性能打分；
- rank 0 调用纯函数 `plan_node_moe_placement` 一次，随后广播不可变 plan；规划错误也
  统一广播，避免其他 worker 卡在 collective；
- 每个 rank 重算 plan checksum、采用自己的 local placement，KV pages 使用所有 rank
  的下界；host/pin budget 按 node-shared aggregate 约束，不再先平均除以 TP size；
- 旧的 2 GiB rank VRAM 差异硬拒绝仅对已接入 joint planner 的 DSV4 offload 降为告警；
  其他 provider 保持原 fail-fast 行为；
- 没有加入自动 layout 搜索、cost model、动态 residency 或 decode-time 通信。

尚未达到阶段退出条件：

- 已完成一次固定 16K KV、cache 768、overlap-off 的 TP4 三重复基线；完整 cache/prefill
  矩阵不再阻塞 P1，后续只在明确验收点按需补测；
- forced plan 的 fixed buffers 尚未成为 mixed route partition 的生产 ABI；当前 forced
  adapter 有意直接调用现有、已验证的 equal-GPU/CPU kernels。进入 P4 前需完成真实
  route worklist 消费和 CUDA Graph capture/replay 验证。

### 2026-08-24：P3 最小显式 weighted TP 接线

已完成：

- 增加 experimental `force_weighted_tp` 与 `--moe-tp-layout`；第一版只接受启动期显式
  rank-order widths，不做自动 layout 搜索、运行期切换或额外 telemetry；
- 布局解析统一生成不可变 `RoutedTpLayout`，校验 TP rank 数、正宽度、256 tile alignment
  和对 routed intermediate size 的精确覆盖；
- DSV4 DS-FP4 bank specs、dummy/serial/parallel loader 以及 w1/w3/w2 checkpoint slice
  全部使用相同 width/offset；原始 safetensors 以外的 TP>1 FTW layout 在 expert allocation
  前 fail-fast；
- `OffloadMoeCache`、prefill buffers、copy descriptors、permanent store 和 dynamic slot bytes
  已确认从 rank-local bank specs/shape 推导，无需新增 weighted 专用分支；routed FP4 kernel
  同样从 bank shape 推导 local width；
- shared expert 保持原 equal TP；routed 与 shared 的本地 partial 相加后仍只进行一次 TP
  all-reduce；增加 `512,512,768,256` offset/slice 和不等宽 partial-sum reference 测试；
- node planner 直接消费该布局和各 rank 实际 shard bytes；forced GPU policy 禁止 planner
  自动加入 CPU layer；启动日志输出每 rank width、offset、expert slot、host/permanent/
  dynamic/KV bytes 和剩余 headroom；
- synthetic copy benchmark 已具备 local width 256/512/768/2048 profile，不需要加载模型。

本轮验证使用 `/home/huoju/work/freetoken-triton-turing-work/.venv`；最终定向测试结果为
`202 passed`，全仓结果为 `1458 passed, 25 skipped`（另有一项既有 FastAPI/Starlette
deprecation warning）。

真实 checkpoint 验收记录：

- 复用 P0 的 equal TP4/16K/cache768/overlap-off 基线，不重跑 equal server；
- 对 weighted `512,512,768,256` 做了两次 startup 尝试，均在 expert bank allocation
  前由 node capacity proof 拒绝，没有进入 CUDA Graph capture 或 decode；
- 第一次暴露出 host budget 在 resident-weight 临时加载峰值期间采样的问题；已改为所有
  rank 在任何权重 materialization 前同步采样一次，消除 rank timing 不确定性；
- 修复后仍只得到 36.48 GiB 默认 host expert budget，而 weighted routed banks 需要
  51.40 GiB aggregate；planner 因此尝试 15.54 GiB/rank permanent placement，并因
  dynamic-cache/KV floor 还差 2.23 GiB/rank 而正确 fail-fast；
- 已停止继续加载模型。下一次验收前先解决默认 host-budget 与这台 123 GiB 节点实际
  稳态可用容量之间的口径差异，或显式给出经确认的 `--moe-host-budget-gb`，不能靠反复
  startup 试错。

尚未达到 P3 完整退出条件：

- 需要在明确验收点仅加载一次 checkpoint，使用固定 16K KV 和同一个简单 eval case，
  对比 `force_equal_tp` 与 `force_weighted_tp` 的输出及端到端性能；
- 当前明确阻塞项是默认 node host-budget 口径：P0 已证明 equal 配置可在该节点保留完整
  host banks（运行期最低 MemAvailable 约 8.83 GiB），但新 planner 在 TP worker 启动后
  只证明出 36.48 GiB，低于 51.40 GiB weighted bank aggregate；
- 尚未完成真实 CUDA Graph capture/replay、cold/warm cache、prefill/decode 的 weighted
  集成验证，因此当前实现保持 experimental forced action，不进入 auto 候选集；
- P4 的 resident EP/full-owner route 仍未开始。
