# MoE 多层资源管理与弹性 Placement 改造计划

状态：核心实施中（Phase 1-5；真实模型矩阵待验证）<br>
目标分支：`feat/moe-resource-placement`（基于上游 `main`，不包含 SM75 兼容）<br>
适用范围：FreeToken MoE offload / hybrid / CPU execution，第一阶段聚焦
DeepSeek-V4 FP4 + TP

## 1. 摘要

FreeToken 论文将完整的 CPU expert pool 作为 source of truth，并把剩余 VRAM
划分为可重建的 GPU expert cache 与 KV cache。该设计的主要优势是：GPU cache
中的内容随时可以从 host banks 恢复，因此 scheduler 可以在 safe point 动态调整
expert slots 与 KV pages，而无需重新加载模型。

当前分支增加的 GPU-only tier 解决了另一类约束：当主机 RAM 不足以保存完整
expert pool 时，把完整 MoE 层流式加载到 GPU protected slots，随后释放对应 host
pages。它提高了可部署性，但当前默认策略会尽可能消耗空闲 VRAM，而且 permanent
slots 与 dynamic cache 共用同一物理分配，导致 GPU-only 启用后不能执行论文所述的
runtime cache rebuild。

本计划将两种机制统一为一个预算驱动的 startup placement planner：

- GPU permanent residency 只用于补足 host RAM 容量缺口，形成不可回收的 VRAM
  最低占用；
- 其余 host-backed experts 继续作为 dynamic GPU cache 的权威来源；
- CUDA pin quota 不足时，在额外 GPU permanent residency 与 CPU-locked residency
  之间选择；
- 扣除 permanent floor 后的 VRAM 继续由论文原有的 elastic manager 在 KV cache
  与 dynamic expert cache 之间调整；
- permanent expert banks 与 dynamic slot cache 物理分离，使 dynamic cache 可以独立
  rebuild。

这不是把所有资源变成可迁移缓存，而是把资源管理明确分成两个阶段：

1. 启动阶段求解不可逆的 authoritative placement；
2. 运行阶段只调整可逆的 elastic pools 和每步 CPU/GPU execution split。

## 2. 背景与设计依据

### 2.1 论文模型

[FreeToken 论文](https://arxiv.org/abs/2608.16157)将个人机器视为 GPU、CPU、
host memory 和 interconnect 组成的统一推理平台。与本计划直接相关的机制包括：

- 完整 expert pool 常驻 CPU，GPU 使用跨层共享的 LRU expert cache；
- decode cache miss 根据实测 PCIe bandwidth 与 CPU host-processing bandwidth，
  通过论文的 `q*` 策略在 GPU cache fill 和 CPU execution 之间分配；
- scheduler safe point 可以调整 GPU expert cache 与 KV cache 的 VRAM 划分；
- cache rebuild 不需要重新加载 experts，因为 CPU pool 始终保存权威副本；
- 无法建立完整 DMA/pinned fast path 时，论文使用纯 CPU MoE 作为部署降级路径。

论文的 elastic memory manager 管理的是可逆缓存，不是不可逆的 model placement。

### 2.2 上游 split-residency

[Issue #55](https://github.com/FlashML-org/FreeToken/issues/55)指出 Windows/WSL
的 WDDM pinned-memory quota 可能低于完整 host expert pool，导致
`cudaHostRegister` 在加载过程中失败。

[PR #112](https://github.com/FlashML-org/FreeToken/pull/112)增加 per-layer host-bank
residency：

- GPU offload layers 使用 `PINNED` host banks；
- CPU layers 使用 `LOCKED` host banks，不消耗 CUDA pin quota；
- `mlock` 失败时降级为 `PAGEABLE`；
- pin-capped 平台可以自动选择足够数量的首尾 CPU layers；
- locked layers 的 decode 走 CPU executor，prefill 使用同步整层拷贝。

该实现解决的是 pin quota，不减少 host RAM 总占用。

### 2.3 当前分支 GPU-only residency

当前分支为 DSV4 FP4 safetensors 增加完整层 GPU-only residency：

- 在 host layer 填充完成后立即复制到 protected GPU slots；
- GPU copy 同步完成后，通过 `MADV_DONTNEED` 释放匿名 host pages；
- permanent slots 不进入 LRU victim 集合；
- TP 下每个 rank 只保存自己的 expert intermediate shard；
- 仍保留至少一层 dynamic cache，prefill overlap 开启时保留两层。

该机制减少 host RAM，但把 protected slots 放在统一 cache allocation 尾部；一旦
host pages 被释放，cache size 就不能安全修改。

### 2.4 上游路线现状

[Roadmap #79](https://github.com/FlashML-org/FreeToken/issues/79)目前没有列出统一
placement planner。Roadmap 评论中已有 VRAM -> RAM -> NVMe 三层 expert storage 的
社区建议，但尚无维护者结论。

[Issue #111](https://github.com/FlashML-org/FreeToken/issues/111)还显示当前
`--moe-cache-auto` 可能把大部分可用 VRAM 分给 expert cache，仅给 KV 留下最低 token
reserve。这说明现有 VRAM 联合预算仍需要更清晰的目标、约束和可观测性。

## 3. 目标与非目标

### 3.1 目标

1. 在 host RAM 不足时，以最少的 GPU permanent layers 补足容量缺口。
2. 在 CUDA pin quota 不足时，自动组合 GPU permanent、host pinned 和 host locked
   tiers。
3. 保留论文的 runtime KV/expert cache elasticity。
4. 让每个 logical expert 始终拥有且仅拥有一个 authoritative copy。
5. 在加载前完成可验证的容量规划，避免加载到一半才 OOM 或 pin 失败。
6. 保留现有 `--moe-gpu-only-layers`、`--moe-cpu-layers` 和 cache 参数的兼容性。
7. 先完成 DSV4 FP4 + TP，再通过 provider capability contract 扩展其他格式。
8. 在启动日志与管理 API 中完整报告预算、placement 和降级原因。

### 3.2 非目标

- 第一阶段不在请求执行期间把 permanent experts 在 RAM 与 VRAM 间迁移。
- 第一阶段不实现 NVMe serving tier。
- 第一阶段不实现 per-expert permanent placement；以完整 MoE layer 为粒度。
- 不改变模型数值语义，不以 expert substitution、skip 或降精度换取容量。
- 不承诺在 permanent VRAM floor 被外部进程挤压后无重启恢复。
- 不一次性支持所有 quant formats、model-owned loaders 和 FTW physical layouts。

## 4. 统一资源模型

### 4.1 术语

代码和 CLI 中应避免使用“pin 到 VRAM”。本项目中的 `pin` 专指
`cudaHostRegister` / pinned host memory。GPU 中不可回收的完整层统一称为
`GPU_PERMANENT` 或 `gpu_resident`。

### 4.2 Authoritative placement tiers

| Tier | 权威副本 | Host RAM | CUDA pin quota | VRAM | Decode |
|---|---|---:|---:|---:|---|
| `GPU_PERMANENT` | GPU | 释放 | 不占用 | 完整层固定占用 | GPU direct |
| `HOST_PINNED` | RAM | 占用 | 占用 | 按需 dynamic slots | GPU offload / hybrid |
| `HOST_LOCKED` | RAM | 占用 | 不占用 | prefill 临时 slots | CPU executor |
| `HOST_PAGEABLE` | RAM | 占用 | 不占用 | prefill 临时 slots | CPU executor，可能 swap |

`HOST_PAGEABLE` 不是主动优化目标，仅是 OS lock 失败后的显式降级状态。

### 4.3 VRAM 布局

```text
VRAM
├── fixed model weights and runtime state
├── permanent expert banks             # startup placement，不可弹性回收
├── elastic pool
│   ├── dynamic expert slot cache       # 可重建
│   └── KV / SWA / recurrent pools      # 可重建或可调整
└── CUDA Graph / activation headroom
```

permanent expert banks 必须与 dynamic expert slot cache 使用不同物理 allocation。
runtime cache rebuild 只能操作 elastic pool。

### 4.4 Source-of-truth invariant

对每个 logical `(layer_id, expert_id)`：

```text
authoritative_copies(layer, expert) == 1
```

- `GPU_PERMANENT`：权威副本位于 immutable GPU bank；host staging pages 已释放；
- 其他 tiers：权威副本位于 host bank；dynamic GPU slots 只是缓存。

dynamic slots 可以随时失效、回收或重建，不影响正确性。

## 5. 容量模型与约束

### 5.1 Planner 输入预算

每个 TP rank 至少需要以下输入：

- `available_vram_bytes`
- `fixed_gpu_bytes`
- `activation_graph_reserve_bytes`
- `host_expert_budget_bytes`
- `pin_budget_bytes`
- `kv_reserve_tokens` / `kv_reserve_bytes`
- `expert_bytes_per_slot`
- `host_bytes_per_layer[layer_id]`
- `gpu_bytes_per_layer[layer_id]`
- `num_experts`、`num_moe_layers`
- `prefill_overlap` 及其 dynamic layer floor
- provider capabilities
- CPU executor viability
- TP / NUMA ownership信息

Host budget 应扣除非 expert 权重、进程运行开销、安全余量和并行 loader transient；不能直接
把 `MemAvailable` 全部交给 expert banks。

### 5.2 Host RAM 缺口

设完整 host expert pool 为 `H_total`，允许给 expert banks 的 host budget 为
`H_budget`：

```text
H_deficit = max(0, H_total - H_budget)
```

选择最小 permanent layer 集合 `P`，满足：

```text
sum(host_bytes_per_layer[l] for l in P) >= H_deficit
```

`HOST_LOCKED` 只减少 pin demand，不减少 `H_total`，因此不能用于解决 host RAM
容量缺口。

### 5.3 Permanent VRAM 可行性

```text
V_permanent = sum(gpu_bytes_per_layer[l] for l in P)

V_elastic = available_vram
            - fixed_gpu_bytes
            - activation_graph_reserve
            - V_permanent
```

必须满足：

```text
V_elastic >= minimum_dynamic_expert_bytes + minimum_kv_bytes
```

其中：

- 有 host-backed GPU layers 时，dynamic cache 至少需要一个完整 expert layer；
- prefill overlap 继续使用统一双缓冲时，至少需要两个完整 layer buffers；
- 全部 layers 都是 `GPU_PERMANENT` 时，dynamic expert floor 可以为零；
- KV reserve 必须是硬约束，不能被 expert cache 贪婪消耗。

如果该不等式不成立，planner 必须在加载前给出不可行报告，而不是静默缩小 KV 到不实用
水平。

### 5.4 Pin quota 缺口

完成 host-capacity placement 后，设剩余 host-backed banks 的 pinned demand 为
`P_demand`：

```text
P_deficit = max(0, P_demand - pin_budget)
```

处理顺序：

1. 如果还有超过 KV/dynamic floor 的可用 VRAM，可以增加 `GPU_PERMANENT` layers；
2. 对剩余缺口选择 `HOST_LOCKED` layers，并把 decode 放到 CPU executor；
3. 如果 CPU executor 不支持当前格式，报告 pin budget 不可行；
4. `mlock` 运行时失败时降级为 `HOST_PAGEABLE` 并明确告警。

是否优先使用额外 GPU permanent 或 CPU locked 应由 policy 决定，而不是写死。

### 5.5 Elastic VRAM 分配

扣除 permanent floor 后，沿用并改进论文的 joint budget solve：

```text
V_elastic = V_dynamic_experts + V_kv + V_other_elastic
```

求解优先级：

1. 满足 KV、dynamic MoE、SWA/GDN 的硬下限；
2. 满足用户显式容量约束；
3. 剩余容量依据 marginal latency value 分配；
4. safe point 允许重新求解并 rebuild dynamic pools；
5. `V_permanent` 不参与 runtime reclaim。

## 6. Placement Planner 设计

### 6.1 数据结构

建议新增纯逻辑模块 `python/freetoken/engine/moe_placement.py`：

```python
class ExpertTier(str, Enum):
    GPU_PERMANENT = "gpu_permanent"
    HOST_PINNED = "host_pinned"
    HOST_LOCKED = "host_locked"
    HOST_PAGEABLE = "host_pageable"


@dataclass(frozen=True)
class MoePlacementPlan:
    layer_tiers: tuple[ExpertTier, ...]
    permanent_layer_ids: tuple[int, ...]
    cpu_layer_ids: frozenset[int]
    permanent_gpu_bytes: int
    retained_host_bytes: int
    pinned_host_bytes: int
    locked_host_bytes: int
    dynamic_cache_slots: int
    kv_pages: int
    prefill_overlap: bool
    constraints: tuple[str, ...]
    decisions: tuple[str, ...]
    warnings: tuple[str, ...]
```

Planner 必须是无 torch/GPU side effect 的纯函数，硬件测量与 checkpoint geometry 由调用方
提供，便于穷举和 property testing。

### 6.2 Provider capability contract

每个 expert provider 声明：

```python
@dataclass(frozen=True)
class ExpertPlacementCapabilities:
    allocation_specs: bool
    gpu_permanent_streaming: bool
    per_layer_host_residency: bool
    cpu_executor_layout: bool
    mixed_cpu_gpu_layout: bool
    ftw_gpu_permanent_streaming: bool
```

Planner 只能生成 provider 明确支持的 tier 组合。显式用户请求不受支持时应直接报错；auto
模式可以降级并记录原因。

### 6.3 求解阶段

```text
Stage 0: 解析显式用户约束与 provider capabilities
Stage 1: 计算 fixed GPU、host、pin、KV 和 dynamic-cache 下限
Stage 2: 用最小 GPU_PERMANENT 集合解决 host RAM 缺口
Stage 3: 解决剩余 pin quota 缺口
Stage 4: 在 elastic VRAM 中联合求解 dynamic expert slots 与 KV pages
Stage 5: 验证所有 invariants，生成 plan 与解释信息
Stage 6: 按 plan 流式加载并验证实际 settled residency
```

### 6.4 Layer 选择策略

第一阶段使用确定性 heuristic：

- host bytes 相同时，默认保留浅层 hash-routed layers 参与 dynamic LRU，把较深层选为
  `GPU_PERMANENT`；
- `HOST_LOCKED` 沿用上游 head+tail 分布，减少对统一 cache 的集中影响；
- 显式 layer ids 优先于 heuristic；
- TP ranks 必须生成相同 logical layer plan。

后续可以使用真实 profiling 数据替代固定 heuristic：

- per-layer active experts；
- dynamic-cache miss rate；
- PCIe bytes avoided；
- GPU-resident latency benefit；
- CPU layer latency与 NUMA bandwidth；
- KV capacity 的 marginal value。

选择目标应是满足容量约束后的 latency 最小化，而不是 GPU-resident layer 数最大化。

### 6.5 不可行报告

失败信息必须包含完整算术，例如：

```text
placement infeasible:
  host expert pool:          137.06 GiB
  host expert budget:         96.00 GiB
  must move to GPU:           41.06 GiB
  permanent GPU required:     10.36 GiB/rank
  elastic VRAM remaining:      8.14 GiB/rank
  required dynamic + KV floor: 9.72 GiB/rank
shortfall:                     1.58 GiB/rank
```

并给出可操作建议：减少 KV reserve、关闭 prefill overlap、增加 TP、降低 graph/request
预算、增加 host RAM，或选择更小/更高量化 checkpoint。

## 7. Runtime 与存储架构改造

### 7.1 分离 permanent banks 与 dynamic cache

当前 protected tail 方案改为：

```text
PermanentExpertStore
  bank_name -> [num_permanent_layers, num_experts, ...]

OffloadMoeCache
  bank_name -> [dynamic_cache_slots, ...]
```

`PermanentExpertStore`：

- 生命周期与 Engine 相同；
- 不参与 LRU、prefill borrowing 或 cache rebuild；
- 保存 logical layer -> resident bank view mapping；
- loader callback 完成同步 H2D 后才允许释放 host staging pages；
- 需要独立的 bytes、load progress 和 health reporting。

`OffloadMoeCache`：

- 只管理 host-backed experts；
- 可以独立 resize/rebuild；
- `slot_for_id` 不为 permanent layers 建立可驱逐 slot mapping；
- permanent layer forward 直接读取 immutable views。

### 7.2 加载顺序

为避免 host RAM 峰值重新达到完整 expert pool：

1. 预先分配 permanent GPU banks；
2. 优先读取包含 `GPU_PERMANENT` layers 的 shards；
3. 每层完成后同步复制到 permanent store；
4. 确认 GPU copy 完成后释放该层 private anonymous pages；
5. 再读取并 settle `HOST_PINNED` / `HOST_LOCKED` layers；
6. 校验实际 settled residency 与 plan；
7. 构建 dynamic copy descriptors 与 CPU executor。

Parallel loader 必须保证即使 shard 跨越多个 tier，也不会在等待 permanent layer 时提前保留所有
host-backed layers。

### 7.3 Prefill

- `GPU_PERMANENT`：直接使用 resident layer views，不发生 H2D；
- `HOST_PINNED`：继续使用 async full-layer double buffer；
- `HOST_LOCKED/PAGEABLE`：同步整层 copy，然后 GPU prefill GEMM；
- 第一阶段可以在存在 locked layer 时全局关闭 overlap；
- 后续应支持 per-layer scheduling，使 pinned layers 保持 overlap、locked layers 只在本层同步。

per-layer overlap 是性能优化，不应阻塞第一阶段容量正确性。

### 7.4 Decode

- `GPU_PERMANENT`：raw expert id 直接映射 resident layer view；
- `HOST_PINNED`：LRU hit/fill 与论文 `q*` hybrid path；
- `HOST_LOCKED/PAGEABLE`：CPU executor；
- hybrid overflow 不得访问已释放 host backing 的 permanent layer；
- CPU/GPU partials 在 TP rank-local layout 内保持兼容，再执行现有 collective。

### 7.5 Runtime rebuild

safe-point rebuild 只改变：

- dynamic expert slots；
- KV pages；
- 其他可重建 state pools。

必须验证：

```text
new_dynamic_slots >= dynamic_floor(plan)
new_kv_pages >= request/admission floor
permanent allocations unchanged
```

当外部 VRAM 压力使总预算低于 fixed weights + permanent floor + activation minimum 时，Engine
不能声称可弹性恢复。第一阶段应拒绝 rebuild，并明确要求释放外部 VRAM 或重启重新规划。

### 7.6 TP 与 NUMA

- placement 使用 logical layer ids，在所有 TP ranks 保持一致；
- bytes 按 rank-local expert shard 计算，日志同时报告 per-rank 与 aggregate；
- host budget 与 pin budget 应按 NUMA node / process ownership 计算；
- CPU executor threads 和 locked banks 绑定同一 NUMA node；
- 某个 rank 不可行时，所有 ranks 必须在大额加载前一致失败。

## 8. CLI 与兼容策略

### 8.1 新增建议参数

```text
--moe-placement auto|elastic|manual
--moe-host-budget-gb auto|N
--moe-pin-budget-gb auto|N
--moe-placement-policy balanced|gpu-first|cpu-first
```

含义：

- `auto`：检测硬件预算并生成完整 plan；
- `elastic`：只有 host RAM/pin quota 需要时才建立 permanent/locked tier；
- `manual`：现有 layer flags 作为精确约束；
- `balanced`：优先保留 KV/dynamic floors，再比较 GPU permanent 与 CPU locked；
- `gpu-first`：pin deficit 优先使用剩余 VRAM；
- `cpu-first`：pin deficit 优先使用 CPU locked。

### 8.2 现有参数映射

| 参数 | 新 planner 中的语义 |
|---|---|
| `--moe-gpu-only-layers auto` | 兼容模式；迁移后表示 planner 可自动使用 GPU permanent，而不是吃满所有剩余 VRAM |
| `--moe-gpu-only-layers N` | 精确要求 N 个 `GPU_PERMANENT` layers |
| `--moe-gpu-only-layers 0` | 禁止 GPU permanent；host RAM 不足时直接报告不可行 |
| `--moe-cpu-layers ...` | 固定 `HOST_LOCKED` / CPU decode layer 集合 |
| `--moe-cache-size N` | dynamic slots，不再包含 permanent slots |
| `--moe-cache-auto` | 只求解 elastic dynamic slots 与 KV pages |
| `FREETOKEN_PIN_BUDGET_GB` | 保留为环境变量兼容入口，CLI 显式值优先 |

`--moe-cache-size` 语义变化需要谨慎迁移。过渡期可以新增
`--moe-dynamic-cache-size`，并对旧参数打印解释后的 permanent/dynamic 拆分。

### 8.3 默认行为

推荐默认规则：

- host RAM 充足且无 pin cap：不建立 permanent tier，保持论文原始模型；
- host RAM 不足：只建立满足缺口的最少 permanent layers；
- pin quota 不足：按 `balanced` policy 组合额外 permanent 与 locked layers；
- 任何自动决策都必须打印预算与原因；
- 不允许 auto 策略静默牺牲 KV reserve。

## 9. 代码改造范围

| 模块 | 改造内容 |
|---|---|
| `engine/moe_placement.py` | 新增纯 planner、数据结构、可行性验证和决策解释 |
| `engine/cache_budget.py` | 保留 elastic pool 算术；移除“GPU-only 吃满额外完整层”的职责 |
| `engine/config.py` | 新预算、policy 和兼容字段 |
| `server/args.py` | CLI、deprecated/compat mapping 和帮助文本 |
| `engine/engine.py` | 一次性生成 plan，按 plan 建 store/cache/CPU executor |
| `moe/permanent_store.py` | 新 immutable GPU expert store |
| `moe/host_banks.py` | 按 plan settle pinned/locked/pageable，报告实际状态 |
| `moe/expert_banks.py` | provider capability、allocation specs、streaming sink contract |
| `moe/offload_cache.py` | 只管理 dynamic slots，删除 protected-tail ownership |
| `models/deepseek_v4/weight.py` | 优先 permanent-layer loading 和 TP-sharded streaming |
| `models/deepseek_v4/moe.py` | per-tier prefill/decode dispatch |
| checkpoint/FTW | 后续增加 TP-aware、per-layer streaming permanent load |
| cache/status API | 输出 permanent/dynamic/KV/host/pin budget 与 plan |
| docs | 更新 CLI、模型说明、容量示例和故障排查 |

## 10. 分阶段实施

### Phase 0：冻结基线与测试夹具

- 固化现有 GPU-only、split-residency、dynamic rebuild 测试；
- 为 DSV4 TP=1/2/4 建立 allocation-only geometry fixtures；
- 记录当前 `/data/models/DeepSeek-V4-Flash-0731` 的启动峰值、steady RAM、per-rank
  VRAM、TTFT 和 decode TPS；
- 定义计划文件中的容量算术 golden cases。

验收：现有行为可重复，所有预算数据可由脚本采集。

### Phase 1：纯 Placement Planner

- 新增 tiers、capabilities、plan 和 feasibility report；
- 接入 host/pin/VRAM/KV/dynamic-floor 算术；
- 先不改变 loader/runtime，仅 shadow-compute 并打印 plan；
- 对随机硬件预算运行 property tests。

验收：planner 无设备副作用；任何成功 plan 满足全部容量 invariants；任何失败包含精确
shortfall。

### Phase 2：分离 PermanentExpertStore

- 把 protected cache tail 迁移到独立 allocation；
- DSV4 permanent forward 改读 immutable store；
- dynamic cache 恢复独立 rebuild 能力；
- 保持当前显式 `--moe-gpu-only-layers N` 行为。

验收：启用 permanent layers 后可以调整 dynamic cache/KV；permanent 权重与映射保持不变；
数值输出与改造前一致。

### Phase 3：Host-budget-driven Placement

- 实现 `--moe-host-budget-gb`；
- GPU permanent 只补足 host deficit；
- 加载器优先 permanent layers 并释放 staging pages；
- 默认 auto 不再使用所有剩余完整层容量。

验收：给定 host budget 时选择最小可行 permanent 集合；加载峰值和 steady host RSS 均低于
预算加显式安全余量。

### Phase 4：统一 Pin-budget 与 CPU Layers

- 将 split-residency 输入统一到 planner；
- pin deficit 可选择额外 GPU permanent 或 host locked；
- settled residency 回写 plan result；
- locked/pageable 层保持 CPU decode 正确性。

验收：WSL/显式 pin budget 下不发生中途 `cudaHostRegister` OOM；pinned bytes 不超过预算；
CPU/GPU mixed output 与全 pinned baseline 一致。

### Phase 5：Elastic Pool 联合重建

- permanent floor 从 runtime budget 中固定扣除；
- dynamic expert/KV cache safe-point rebuild；
- admission 拒绝超过 KV capacity 的请求，避免 Issue #111 的无限等待；
- cache status API 报告完整资源拓扑。

验收：多次扩大/缩小 KV 与 dynamic expert cache 后输出正确、无 host reload、无 permanent
mapping 变化、无请求永久排队。

### Phase 6：性能优化与泛化

- per-layer prefill overlap；
- 使用 routing/miss profiling 选择 permanent/locked layers；
- 扩展 NVFP4、MXFP4、BF16、GGUF providers；
- 设计 FTW vNext 的 TP-aware per-layer streaming；
- 评估 NVMe authoritative tier，但不与前述阶段绑定。

验收：每个 provider 只在 capability contract 完整时启用对应 tier；auto 模式不产生隐式布局
转换。

## 11. 测试与验证矩阵

### 11.1 纯单元测试

- host deficit 为 0 / 刚好一层 / 非整层倍数 / 全模型；
- permanent VRAM 刚好满足、差 1 byte、全层常驻；
- pin deficit 分别由 GPU、CPU、混合方式解决；
- KV 与 dynamic floor 永远是硬约束；
- provider capability 不支持时 auto 降级、explicit 拒绝；
- TP per-rank 与 aggregate bytes 一致；
- 随机预算下 invariant property tests；
- 不可行报告中的算术可复算。

### 11.2 CUDA 组件测试

- permanent layer load、direct views 和数值指纹；
- permanent allocation 与 dynamic cache 地址/生命周期分离；
- dynamic rebuild 后 permanent 内容不变；
- LRU 永不访问 permanent store 的 host source；
- reset、graph capture 和 replay 不改变 authoritative mapping；
- pageable/locked materialize 不进入 fused UVA copy path。

### 11.3 集成测试

组合矩阵：

| Permanent | Pinned | Locked | Overlap | Rebuild | 预期 |
|---:|---:|---:|---:|---:|---|
| 0 | all | 0 | on | yes | 论文原始路径 |
| some | rest | 0 | on/off | yes | RAM deficit 路径 |
| 0 | some | some | off | yes | split-residency 路径 |
| some | some | some | off | yes | 完整统一路径 |
| all | 0 | 0 | off | dynamic=0 | 全 GPU authoritative |

每种组合验证 prefill、decode、cache reset、请求切换、TP collective 和 engine shutdown。

### 11.4 故障注入

- `cudaHostRegister` 达到预算边界；
- `mlock` / RLIMIT_MEMLOCK 失败；
- permanent H2D copy 中断；
- 某 TP rank budget 不一致；
- runtime rebuild 低于 permanent + elastic floor；
- parallel loader shard 跨 permanent/host tiers；
- 加载中 backend worker 异常退出，确认不会遗留大额 pinned/GPU allocation。

### 11.5 真实模型验证

优先使用 `/data/models` 中已有权重，不联网下载：

- DeepSeek-V4-Flash DSV4 FP4，TP=1/2/4；
- 若存在，补充 Qwen/MiniMax NVFP4 验证 split-residency provider 泛化；
- greedy 固定 prompts 比较 token 输出；
- sampled 请求比较统计一致性，不只跑启动 smoke test；
- 记录 cold start、peak/steady RAM、pinned bytes、per-rank VRAM、KV tokens、TTFT、prefill
  TPS、decode TPS 和 cache miss rate。

## 12. 可观测性

启动时必须输出一张统一表：

```text
MoE placement plan
  expert pool:          137.06 GiB aggregate
  GPU permanent:         16 layers / 51.00 GiB aggregate
  host retained:         27 layers / 86.06 GiB aggregate
    pinned:              19 layers / 60.56 GiB
    locked:               8 layers / 25.50 GiB
  dynamic GPU cache:    256 slots / 0.80 GiB per rank
  KV reserve:         65,536 tokens / 4.20 GiB per rank
  prefill overlap:      disabled (locked layers present)
  policy reason:        host deficit + pin quota
```

`ft ctl cache` / stats API 增加：

- permanent bytes 与 layer ids；
- dynamic slots、occupancy 和 rebuild floor；
- KV pages/tokens 与 admission limit；
- host retained/pinned/locked/pageable bytes；
- planner constraints、warnings 和实际 settled 差异；
- TP per-rank/aggregate 视图。

日志必须区分 `requested plan` 与 `actual residency`，尤其是 mlock 失败后的 pageable 降级。

## 13. 风险与缓解

### 13.1 Permanent floor 降低运行时弹性

风险：外部应用抢占 VRAM 后，Engine 无法回收 permanent experts。

缓解：只用最少 permanent layers 解决 host deficit；预留更保守 headroom；低于 floor 时明确
拒绝 rebuild，而不是 OOM 后继续运行。

### 13.2 Host 加载峰值超预算

风险：虽然 steady state 满足预算，但加载顺序仍暂时物化完整 pool。

缓解：permanent layers 优先、逐层同步 H2D、private mmap + `MADV_DONTNEED`、串行低内存
fallback、记录 peak RSS。

### 13.3 Dynamic cache 性能下降

风险：permanent floor 挤压 dynamic slots 或 KV，导致 miss rate 或 context capacity 恶化。

缓解：KV/dynamic floors 作为硬约束；GPU permanent 只补容量缺口；报告 marginal tradeoff；
禁止 auto 吃满 VRAM。

### 13.4 静态 layer heuristic 不适合所有 workload

风险：深层 permanent、首尾 locked 的固定选择可能不是最优。

缓解：保持 correctness 与 feasibility 独立于 ranking；收集 per-layer stats；后续允许 profile 驱动
和 explicit override。

### 13.5 Provider layout 不兼容

风险：CPU 与 GPU backend 使用不同 physical bank layouts，无法对不同 layers 混合。

缓解：capability contract；第一阶段仅 DSV4 `ds_fp4`；不支持时明确拒绝，不进行隐式重排。

### 13.6 CLI 语义变化

风险：用户把 `--moe-cache-size` 理解为包含 permanent slots 的总量。

缓解：过渡期增加 dynamic-specific 参数；启动日志打印拆分；旧参数映射附带 warning；在 major
版本再清理。

## 14. 上游协作策略

建议把改造描述为论文 §3.3 的保守扩展，而不是替换其架构：

1. 保留 host-backed expert 的 source-of-truth 与 dynamic rebuild 语义；
2. 仅在 host capacity 不可行时引入最小 GPU authoritative tier；
3. 把 split-residency 作为同一 planner 的 pin-budget 解；
4. 将 permanent store 与 dynamic cache 分离，避免破坏现有 runtime elasticity；
5. 分 PR 提交：pure planner -> storage separation -> host budget -> pin integration ->
   provider expansion；
6. 新功能先开 issue，并按 `CONTRIBUTING.md` 与维护者确认设计范围。

上游提案应包含至少三组数据：

- 论文原始 all-host 配置无性能回归；
- host RAM 不足配置从无法启动变为可启动；
- permanent tier 启用后 dynamic cache/KV rebuild 仍可工作。

## 15. 开放问题

1. `--moe-cache-size` 最终表示 dynamic slots 还是 permanent + dynamic 总 slots？
2. Host budget 的自动安全余量应按固定 GiB、RAM 比例还是加载器 transient estimate？
3. Pin deficit 下 `balanced` policy 如何量化“增加 permanent layer”与“增加 CPU layer”的
   latency tradeoff？
4. permanent layer ranking 使用离线默认、启动 benchmark，还是历史 routing profile？
5. per-layer prefill overlap 是否需要进入第一版，还是先全局关闭？
6. FTW vNext 是否把 expert banks 按 TP size 与 layer 独立存储？
7. runtime 是否允许在 RAM 重新可用时把 GPU permanent layer demote 回 host？
8. 多请求场景下 KV marginal value 与 expert-cache marginal value 如何估计？
9. `HOST_PAGEABLE` 是否允许生产运行，还是只作为带强告警的最后降级？
10. NVMe tier 将作为 authoritative storage、cold cache，还是仅用于 engine restart 加速？

## 16. 完成定义

本改造在满足以下条件后视为完成：

- 一个统一 planner 同时求解 host RAM、pin quota、permanent VRAM、dynamic expert cache 与
  KV reserve；
- GPU permanent allocation 与 dynamic cache 物理分离；
- host RAM 不足时仅选择最小必要 permanent layer 集合；
- permanent tier 启用时仍可在线 rebuild dynamic expert/KV pools；
- split-residency 不再由独立分支逻辑求解；
- 所有成功 plan 都能在加载前通过容量证明，并在加载后验证 actual residency；
- DSV4 FP4 TP=1/2/4 真实 checkpoint 验证通过；
- 原始 all-host/pinned 路径无数值和显著性能回归；
- API、CLI、日志和文档能够解释每一项自动决策；
- 对不支持的 provider/format 给出明确、可操作的错误，而不是静默降级或中途 OOM。

## 参考资料

- [FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution](https://arxiv.org/abs/2608.16157)
- [FreeToken Roadmap 2026, Issue #79](https://github.com/FlashML-org/FreeToken/issues/79)
- [Windows/WDDM pinned-memory quota, Issue #55](https://github.com/FlashML-org/FreeToken/issues/55)
- [Per-layer host-bank residency, PR #112](https://github.com/FlashML-org/FreeToken/pull/112)
- [Partial-pin serving proposal, PR #27](https://github.com/FlashML-org/FreeToken/pull/27)
- [KV/expert auto-budget issue, Issue #111](https://github.com/FlashML-org/FreeToken/issues/111)
