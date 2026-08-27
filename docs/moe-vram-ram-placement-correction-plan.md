# MoE VRAM + RAM 联合 Placement 修正计划

> 状态：C0–C5 已实施；C4 真实 checkpoint 验收通过  
> 记录日期：2026-08-27  
> 目标分支：`work/moe-heterogeneous-upstream-merge`  
> 首个目标平台：4 × NVIDIA TITAN RTX（SM75），DeepSeek-V4-Flash-0731  
> 相关文档：
> [MoE 异构混合执行分阶段实施计划](moe-hybrid-execution-implementation-plan.md)、
> [MoE 多层资源管理与弹性 Placement 改造计划](moe-resource-placement-plan.md)、
> [异构 PCIe 环境下的 MoE 并行与专家执行设计](moe-heterogeneous-pcie-parallelism-design.md)

## 0. 实施记录

2026-08-27 已完成以下代码落地：

- fixed KV 使用 `kv_target_pages`，auto KV 使用 `kv_min_pages`；node plan 与每个 rank plan
  统一为相同页数，engine 在创建 KV pool 前校验 plan/runtime 一致；
- node planner 已删除 proportional budget apportionment，改为 rank candidate frontier 与
  node-shared host/pin 约束的多选择组合；weighted regression 得到
  `15/15/9/33` 个 permanent layers，retained host 精确落在 aggregate budget；
- node plan checksum 覆盖 rank-specific tiers、fixed/dynamic/KV/permanent/headroom bytes、
  aggregate pinned/locked/permanent bytes、KV mode 和 runtime disk 标记；
- loader/cache 按本 rank 的 permanent layer IDs 工作；同一 logical layer 的 mixed
  rank residency、host source 丢弃和 dynamic rebuild 已有 CPU 单元回归；
- normal benchmark 明确排除 disk-backed；emergency 路径增加 refill reads/experts/bytes/time
  telemetry，benchmark JSON 从 server log 保存 placement checksum、KV plan 和 storage stats。

C4 的首次启动尝试在 expert allocation 前被一项错误校验拒绝：fixed 128 pages 被拿去与
auto-KV reserve 485 pages 比较。该校验已修正为只使用 pool intrinsic admission floor
（本配置为 16 pages），并增加 `128 >= 16` 与 `15 < 16` 回归。原始失败日志为
`/tmp/bench-serve-offload-0xsyqrvx.log`，没有生成 benchmark JSON，四卡显存均恢复为 1 MiB；
随后经明确授权只重试一次相同 C4 命令。

授权后的真实 checkpoint 验收已通过：DeepSeek-V4-Flash-0731、TP4、weighted layout
`512,512,768,256`、fixed 16K KV、cache 768、overlap off、CUDA Graph on、normal no-disk
模式成功进入 serving 并完成一轮 warm-up 和三次 64-token decode。权威 plan checksum 为
`f9a46cd7d37a`；同次 snapshot 的 safe expert host budget 为 90.62 GiB，实际 retained host
为 90.45 GiB（pinned 90.45 GiB、locked 0），permanent GPU 为 46.62 GiB。每 rank 计划为：

| rank | width | permanent layers | fixed | permanent | host retained | dynamic | fixed KV | headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 15 | 8.08 GiB | 11.95 GiB | 22.31 GiB | 2.39 GiB | 0.55 GiB | 0.16 GiB |
| 1 | 512 | 15 | 8.08 GiB | 11.95 GiB | 22.31 GiB | 2.39 GiB | 0.55 GiB | 0.16 GiB |
| 2 | 768 | 8 | 8.08 GiB | 9.56 GiB | 41.84 GiB | 3.59 GiB | 0.55 GiB | 1.35 GiB |
| 3 | 256 | 33 | 8.08 GiB | 13.15 GiB | 3.98 GiB | 1.20 GiB | 0.55 GiB | 0.16 GiB |

rank 0/1 permanent layer IDs 为 28–42，rank 2 为 35–42，rank 3 为 10–42；loader 完成后
runtime 明确确认 actual residency 与 plan 一致。plan 与 `/v1/cache/status` 均报告 fixed
128 pages / 16384 tokens；运行期 expert disk read/refill 均为 0。startup/serving 资源采样
的最低 `MemAvailable` 为 5.97 GiB，swap free 无下降（结束时增加 4 KiB），无 host/GPU OOM、
collective timeout 或 Xid；退出后四卡均回到 1 MiB。

三次 measured decode 中位数为 10.742 tok/s、93.097 ms/token、TTFT 2.243 s、event p99
143.852 ms，轮间吞吐极差 0.005 tok/s，三次 greedy SHA1 均为 `3cc47752354e`。相对同 prompt、
16K KV、cache 768、overlap-off、CUDA Graph-on 的 no-disk equal-TP 基线，吞吐提升 42.77%，
TTFT 降低 17.42%，p99 降低 29.78%；weighted/equal hash 差异符合既有 FP16 partial
reduction grouping 的数值容差口径。原始结果位于
`benchmarks/results/moe-placement-c4-20260827/weighted-tp4-cache768-kv16k.jsonl`，server log
为 `/tmp/bench-serve-offload-6g6nqfyk.log`。

最终工作树使用项目 venv 全仓复测：`1480 passed, 25 skipped`，仅有一项既有
FastAPI/Starlette deprecation warning。

## 1. 决策摘要

本修正恢复最初的资源目标：当扣除 non-expert weights、运行时固定开销、安全余量、
dynamic expert cache 和用户要求的 KV cache 后，节点可用 VRAM 与 RAM 足以容纳全部
routed expert shards 时，运行期权威专家权重必须只位于 VRAM 或 RAM，不得读取 SSD。

原始 safetensors 在正常模式中只承担 checkpoint/startup source，不是 serving tier。
启动 loader 可以从 SSD 流式读取一层，然后按最终计划把 rank-local shard 固化到：

- `GPU_PERMANENT`：复制完成后释放该 shard 的 host pages；
- `HOST_PINNED` / `HOST_LOCKED`：保留 RAM 权威副本，运行时按现有 dynamic cache 或
  CPU executor 使用。

`--moe-disk-backed` 只保留为用户显式选择的低内存应急/诊断模式：

- 不进入 `auto` placement 候选集；
- 不作为 P3 weighted TP 的容量或性能验收路径；
- 不允许 planner 在 VRAM + RAM 可行时自动降级到它；
- 未显式授权时，VRAM + RAM 不可行必须在加载前给出完整算术并失败。

本轮需要修正的核心问题有两个：

1. node planner 不能先按 shard 大小把共享 RAM budget 切给各 rank，再独立求解；它必须
   在节点级联合选择每个 `(rank, layer)` 的权威 residency，允许不同 rank 永久驻留不同
   数量、不同集合的 layer shards；
2. 显式 `--num-tokens` / `--num-pages` 必须是 KV 硬容量，不能在 placement plan 中被当作
   仅有的 reserve floor 后继续吃满剩余 VRAM。

## 2. 问题复盘

### 2.1 实际发生的路径

weighted layout `512,512,768,256` 的完整 routed-expert host backing 为约
137.06 GiB。早期 planner 在没有显式 host budget 时把完整 host pool 本身当作可用预算，
因此选择了：

```text
GPU permanent:  0 layers
host retained:  137.06 GiB aggregate
```

该方案没有执行 VRAM + RAM 联合 placement，而是直接尝试把全部权威专家 shards 留在
RAM。加载期间 scope 峰值达到 119.5 GiB，随后被 host OOM killer 终止。

OOM 后增加的 `--moe-disk-backed` 将所有逻辑层映射到一个 rank-local staging layer，
并明确禁止 GPU permanent placement。该路径解决了内存上界和正确性 smoke，但使每次
GPU cache miss 都同步读取 safetensors，18-token prefill 只有约 0.08–0.18 token/s。

因此，disk-backed 是绕过 placement 问题的实验性路径，不是“VRAM + RAM 确实不足”的
证明，也不能替代本计划要求的联合容量求解。

### 2.2 首个平台的容量事实

DSV4 DS-FP4 每个未分片 expert 约 12.75 MiB；一层 256 个 experts 的 aggregate payload
为 3.1875 GiB。weighted TP 的 rank-local 几何为：

| rank | width | 每层 shard | 43 层 host pool | cache 768 |
|---:|---:|---:|---:|---:|
| 0 | 512 | 0.796875 GiB | 34.27 GiB | 2.390625 GiB |
| 1 | 512 | 0.796875 GiB | 34.27 GiB | 2.390625 GiB |
| 2 | 768 | 1.1953125 GiB | 51.40 GiB | 3.5859375 GiB |
| 3 | 256 | 0.3984375 GiB | 17.13 GiB | 1.1953125 GiB |
| aggregate | 2048 | 3.1875 GiB | 137.06 GiB | 9.5625 GiB |

一次实机采样给出的安全 expert RAM budget 约为 90.80 GiB，因此至少需要从 RAM 移走：

```text
137.06 - 90.80 = 46.26 GiB
```

在 `memory_ratio=1.0`、cache 768、固定 16K KV 的已有日志几何下，粗略的 rank-local
permanent layer 上限为：

```text
rank 0 / 1 / 2 / 3:  15 / 15 / 9 / 33 layers
aggregate host bytes removable: about 47.81 GiB
```

这说明 aggregate 容量存在可行解，但它要求非对称 residency。按 host shard 比例切分
RAM budget 会近似强迫每个 rank 都移动 15 层；rank 2 只能容纳约 9 层，因此旧的局部
求解会错误报告不可行。

上述数字只用于说明已有失败的结构原因，不作为通用硬编码。正式 planner 必须使用每次
启动采集到的整数 byte geometry、安全余量和实际 KV cost 重新证明。

### 2.3 固定 KV 被错误表示

已有 16K KV 运行使用 DSV4 128-token pages，因此显式目标是 128 pages。placement 日志
却报告：

```text
KV reserve/plan: 128/2683 pages
```

当前 local solver 在满足 reserve 后仍把所有剩余 VRAM换算成 `kv_pages`。即使 runtime
保留了显式 `num_page_override=128`，返回的 plan、headroom 日志和后续容量判断仍混入了
“自动最大 KV”语义，产生 rank 2 KV 约 11.46 GiB、headroom 为零的误导性结论。

固定 KV、KV 下限和自动 KV 是三个不同概念，必须在输入和 plan 中显式区分。

## 3. 不可破坏的约束

### 3.1 权威副本

对每个 rank-local logical shard `(rank, layer, expert)`：

```text
authoritative_copies(rank, layer, expert) == 1
```

合法权威 tier 为 `GPU_PERMANENT`、`HOST_PINNED`、`HOST_LOCKED` 和显式允许时的
`HOST_PAGEABLE`。正常 serving plan 中不包含 disk tier。

GPU dynamic cache 不是权威副本，可以随时从对应 host tier 重建。若某个 rank-layer
是 `GPU_PERMANENT`，其 host pages 必须在 GPU copy 完成后释放，且该 rank-layer 不进入
dynamic cache victim/source 集合。

### 3.2 Rank 非对称合法

同一个 logical MoE layer 在不同 TP ranks 上可以采用不同 tier。例如：

```text
layer 17:
  rank 0 -> HOST_PINNED
  rank 1 -> GPU_PERMANENT
  rank 2 -> HOST_PINNED
  rank 3 -> GPU_PERMANENT
```

每个 rank 仍计算固定 intermediate slice，最后执行同一次 `[tokens, hidden]` all-reduce。
residency 不对称不能改变 routed intermediate widths、router 结果或 collective 顺序。

### 3.3 容量约束

令 `x[r,l] = 1` 表示 rank `r` 的 layer `l` shard 位于 `GPU_PERMANENT`，否则保留在
host tier。`H[r,l]` 和 `G[r,l]` 分别是它的 host/GPU bytes。

节点 host 约束：

```text
sum(r,l, (1 - x[r,l]) * H[r,l]) <= host_expert_budget
```

每 rank VRAM 约束：

```text
fixed_gpu[r]
+ activation_graph_reserve[r]
+ sum(l, x[r,l] * G[r,l])
+ dynamic_cache_bytes[r]
+ kv_bytes[r]
<= available_vram[r]
```

pin quota 是 node-shared aggregate 约束；它不能通过先平均或按 shard 比例硬切分而变成
不必要的 per-rank blocker。

### 3.4 KV 语义

- 显式 `--num-tokens N`：解析为固定 `kv_target_pages`，plan 和 runtime 必须都精确使用；
- 显式 `--num-pages N`：同样是固定 `kv_target_pages`；
- `--kv-reserve-tokens N`：只在 KV auto 模式中提供硬下限；
- 无显式 target 且启用 auto：先完成权威 expert placement，再在剩余 elastic VRAM 中
  联合选择 dynamic slots 与 KV pages；
- 日志必须分别输出 `fixed`、`minimum` 或 `auto-selected`，不得再使用含混的
  `reserve/plan` 表达固定配置。

### 3.5 SSD 边界

正常模式允许在 startup 读取 checkpoint；engine ready 后：

```text
runtime_expert_disk_reads == 0
```

只有显式 `--moe-disk-backed` 可以改变该约束。该 flag 不得由 `auto`、OOM catch 或
planner fallback 隐式打开。

## 4. 新的节点级求解方法

### 4.1 删除 proportional budget apportionment

`plan_node_moe_placement` 不再调用 `_apportion_budget()` 把 host/pin budget预分给 ranks。
rank-local solver 只负责生成满足本 rank VRAM、dynamic floor、fixed KV 和 capability 的
候选 frontier；node solver 负责组合候选并验证 aggregate host/pin 约束。

### 4.2 Rank-local candidate frontier

对每个 rank 生成一组候选状态：

```text
Candidate:
  permanent_layer_ids
  permanent_gpu_bytes
  host_bytes_removed
  retained_pinned_bytes
  retained_locked_bytes
  dynamic_cache_slots
  kv_pages
  vram_headroom_bytes
```

DSV4 第一版每个 rank 内的 layer shard 等大，可以为 `k=0..num_layers` 生成确定性的
prefix candidates，无需引入通用 ILP。layer ID 顺序沿用既有稳定 heuristic；容量求解
首先选择数量，具体 ID 用独立、可测试的 deterministic order 决定。

若后续 provider 的 per-layer bytes 不同，先生成 Pareto frontier，删除同时满足“使用更多
VRAM、释放更少 host、headroom 更小”的被支配候选；不要恢复 proportional split。

### 4.3 Node frontier 组合

node solver 逐 rank 合并 candidate frontiers并持续做 Pareto pruning。最终选择满足以下
条件的 deterministic 最优解：

1. aggregate retained host/pinned bytes 不超过预算；
2. 每个 rank 满足 VRAM、dynamic 和 KV 硬约束；
3. 不使用 disk tier；
4. 最小化超出 host deficit 的 permanent GPU bytes；
5. 最大化最小 rank headroom；
6. 再以 PCIe benefit、既有 layer order 和 rank id 做稳定 tie-break。

第一版不需要根据运行时 route 频率做复杂 placement。PCIe tie-break 只能在容量等价的候选
之间使用，不能破坏容量证明或制造不可复现的计划。

### 4.4 显式 policy

- `--moe-gpu-only-layers auto`：允许 node solver为每个 rank选择不同数量和集合；
- `--moe-gpu-only-layers N`：继续表示用户要求的严格策略。兼容期内将其定义为每 rank
  精确 N 层；若以后需要 node aggregate 数量，新增参数而不是静默改变语义；
- `--moe-gpu-only-layers 0`：禁止 permanent；host 不足时直接报告不可行；
- `force_equal_tp` / `force_weighted_tp`：固定 execution layout，不禁止自动
  `GPU_PERMANENT` authoritative placement；
- `--moe-placement manual`：不添加用户未指定的 permanent/CPU layers；
- `--moe-disk-backed`：独立的显式 emergency mode，不与 normal node solver混合。

## 5. 数据结构与接口修改

### 5.1 Placement inputs

在 `MoePlacementInputs` 中把 KV 字段拆清楚，建议最小接口为：

```python
kv_min_pages: int
kv_target_pages: int | None       # 非 None 表示 fixed
kv_page_bytes: int
dynamic_cache_slots: int | None   # 非 None 表示 fixed
```

不再用一个 `kv_reserve_pages` 同时表达 fixed target 和 auto floor。

`NodeMoePlacementInputs` 保留一个 node-shared `host_expert_budget_bytes` 和
`pin_budget_bytes`，不再产生 rank-local host/pin budget占位值。

### 5.2 Placement plan

`NodeMoePlacementPlan` 必须明确包含：

- 每个 rank 的 `permanent_layer_ids` 和完整 `layer_tiers`；
- per-rank fixed/dynamic/KV/permanent/headroom bytes；
- aggregate retained host、pinned、locked 和 permanent GPU bytes；
- `kv_mode: fixed | auto` 及其输入/输出 pages；
- 是否含 runtime disk tier；正常 plan 必须为 false；
- checksum 覆盖所有以上字段。

不能再用 rank 0 的 permanent layer 数代表整节点，也不能只打印 rank 0 的 pinned bytes。

### 5.3 Loader/runtime contract

现有 `ServingLayerPipeline`、`PermanentExpertStore` 和 rank-local
`gpu_resident_layer_ids` 可以继续作为落地机制，但必须验证以下新组合：

- 各 rank 的 `gpu_resident_layer_ids` 不相同；
- 某层部分 ranks 使用 raw expert ID/permanent views，其他 ranks 使用 LRU slot IDs；
- 每个 rank 的 routed partial shape 仍为 `[tokens, hidden]`；
- 所有 ranks 每层恰好执行一次、顺序一致的 all-reduce；
- CUDA Graph capture/replay 不要求各 rank 在 collective 前拥有相同的本地 kernel 序列，
  但 collective 序列必须完全一致；
- permanent copy完成前不得丢弃 host pages，失败清理不能留下半有效 authoritative state。

## 6. 分阶段实施

### C0：冻结错误验收口径

1. 在原 P3 实施记录中把 disk-backed smoke 标记为 bounded-memory emergency path，不是
   weighted TP 容量/性能退出条件。
2. P3 正式 benchmark 命令删除 `--moe-disk-backed`。
3. 保留当前 disk tests，防止应急路径损坏，但不以其通过代表 normal placement通过。
4. 明确比较使用相同的 `memory_ratio`、KV、dynamic cache、graph 和 overlap 设置。

退出条件：文档和 benchmark harness 不再把 disk-backed 结果混入 normal P3 基线。

### C1：修正 fixed KV contract

1. CLI/config 将 `num_token_override` 在 page size 最终确定后转换为 fixed pages；
2. placement input区分 `kv_target_pages` 与 `kv_min_pages`；
3. fixed 模式下 local/node plan 的 `kv_pages` 必须等于 target；
4. 只有 auto 模式可以把剩余 VRAM分给更多 KV pages；
5. cache pool创建后校验 actual pages 与 placement plan 完全一致。

退出条件：`--num-tokens 16384` 在 DSV4 上从 plan 到 runtime 始终为 128 pages，日志不再
出现 `128/2683`。

### C2：实现 node-global asymmetric solver

1. 删除 host/pin proportional apportionment；
2. 生成每 rank capacity frontier；
3. node 级组合并 Pareto prune；
4. 输出 rank-specific layer tier vectors；
5. 在不可行错误中同时报告 aggregate deficit 和每 rank frontier上限；
6. 保持 planner 为无 torch/CUDA/filesystem 副作用的纯函数。

退出条件：新增 regression fixture 必须证明“proportional local solve失败、node aggregate
solve成功”，并得到不同 rank permanent layer counts。

### C3：接通非对称 permanent runtime

1. 每个 worker只使用自己的 rank plan建立 `PermanentExpertStore`；
2. loader按本 rank tier流式保存到 GPU 或 RAM；
3. 校验 rank-specific `gpu_resident_layer_ids` 与实际 store一致；
4. 验证同层 mixed residency 的 prefill、decode、cache reset 和 all-reduce；
5. 验证 CUDA Graph capture/replay；graph不支持的组合必须在模型加载前明确失败。

退出条件：至少一个 TP4 测试中四个 rank 的 permanent layer集合不同，输出与
all-host reference 在既有数值容差内一致。

### C4：恢复真实 checkpoint 验收

只启动一次真实 checkpoint，固定：

```text
model:             DeepSeek-V4-Flash-0731
TP:                4
layout:            512,512,768,256
KV:                16K fixed
dynamic cache:     768（若容量证明需要调整，另建 case，不静默改变）
prefill overlap:   off（先与既有基线一致）
execution policy:  force_weighted_tp
disk-backed:       false
```

验收必须记录：

- 启动前 budget snapshot 和 reserve；
- per-rank fixed/permanent/dynamic/KV/headroom bytes；
- per-rank permanent layer IDs；
- aggregate retained/pinned/locked RAM；
- startup peak RAM、稳态 RAM、swap delta；
- engine ready 后 checkpoint 文件的读字节增量；
- 固定 prompt 的 TTFT、prefill token/s、decode token/s 和 output hash；
- equal TP 与 weighted TP 使用完全相同的测量参数。

退出条件见第 8 节。

### C5：收紧 SSD emergency mode

1. CLI help 和启动日志明确标记“runtime expert reads from disk; throughput may be unusable”；
2. stats 暴露 disk refill experts/bytes/time；
3. normal planner plan 中发现 disk tier视为内部错误；
4. `auto` 永不选择 disk；
5. 若长期不维护该模式，在 VRAM + RAM 路径稳定后单独决定删除，不与本修正绑定。

## 7. 测试计划

### 7.1 纯 planner 单元测试

- aggregate VRAM + RAM 可行，但 proportional rank split不可行；
- rank 2 宽 shard受限，rank 3 使用更多 permanent layers补足 aggregate host deficit；
- equal TP仍可产生对称 plan；
- fixed 16K KV精确得到128 pages；
- auto KV只在 authoritative placement完成后使用剩余 VRAM；
- fixed cache 低于 dynamic floor时失败；
- host deficit超过所有 rank frontier总和时输出完整 shortfall；
- `gpu_only_layers=0`、严格 N、manual policy保持原语义；
- candidate/frontier顺序变化不改变最终 checksum；
- 非等大 layer bytes的 Pareto pruning不丢失可行解。

### 7.2 Loader/cache 测试

- 两个模拟 ranks 对同一 layer选择不同 tier；
- GPU permanent rank释放 host pages，host rank保留 source；
- dynamic cache只为 host-backed rank-layer分配/拷贝；
- reset/rebuild不访问 permanent rows，也不要求所有 ranks tier相同；
- copy或加载中途失败时，资源清理后不存在双权威副本；
- normal mode不构造 `Dsfp4DiskExpertSource`。

### 7.3 分布式数值测试

- TP4 asymmetric residency prefill对齐 unsharded/equal reference；
- decode 1、8、64 steps；
- cache cold/warm、重复 experts和全 miss；
- greedy output deterministic；weighted与equal按数值容差验收，不强求逐 token hash相同；
- CUDA Graph capture/replay及 eager各一组；
- all ranks collective调用次数和顺序一致。

### 7.4 实机资源测试

- `memory_ratio=1.0` 和保守 reserve配置分别给出 plan，不混写结果；
- cache 256/768、KV 16K fixed矩阵只做 planner dry-run，选择一个 case加载 checkpoint；
- startup OOM/swap、host page释放和VRAM峰值；
- ready 后对模型文件监测零读取；
- 相同 prompt重复请求不产生 checkpoint page faults；
- server shutdown后四卡和host allocations恢复。

## 8. 阶段退出条件

以下条件必须全部满足，才能恢复 P3 完整验收并继续 P4：

1. 使用 `force_weighted_tp` 且不传 `--moe-disk-backed` 成功进入 serving；
2. plan证明 aggregate retained expert RAM不超过同次启动采样的安全 budget；
3. 所有 ranks分别满足 fixed + permanent + dynamic + fixed KV + reserve 的 VRAM约束；
4. 固定 16K KV 在 plan 和 runtime中均为128 pages；
5. 至少两个 ranks的 permanent layer数量或集合不同，证明非对称 placement真正生效；
6. engine ready 后 routed expert checkpoint读取量为零；
7. swap无增长、无 host OOM、无 GPU OOM、无 collective timeout或Xid；
8. 输出满足现有 weighted TP数值容差，重复 greedy运行确定；
9. 相同参数下，TTFT不得超过 no-disk equal-TP baseline 的2倍；若未达到，必须有分项
   telemetry证明瓶颈，不得回退到 disk-backed完成验收；
10. 最终结果包含可复现命令、commit、plan checksum和原始 JSON/log路径。

## 9. 可观测性要求

启动日志改为一张真正的 node plan表，而不是 rank 0摘要：

```text
MoE authoritative placement
  mode: VRAM+RAM (runtime disk disabled)
  host expert budget: 90.80 GiB aggregate
  retained host:       90.44 GiB aggregate
  permanent GPU:       46.62 GiB aggregate
  KV:                  fixed 128 pages / 16K tokens

  rank  width  permanent-layers  permanent  host-retained  dynamic  KV   headroom
  0     512    ...               ...        ...            ...      ...  ...
  1     512    ...               ...        ...            ...      ...  ...
  2     768    ...               ...        ...            ...      ...  ...
  3     256    ...               ...        ...            ...      ...  ...
```

不可行报告必须区分：

- aggregate VRAM + RAM确实不足；
- 某 rank VRAM不足但其他 ranks仍有可转移容量；
- fixed dynamic cache或fixed KV使方案不可行；
- activation/graph reserve使方案不可行；
- provider不支持非对称 permanent streaming；
- 用户显式 policy阻止自动 placement。

报告建议必须对应真实约束，例如降低 dynamic cache、降低 fixed KV、关闭/缩小 graph、
调整 memory ratio、增加 RAM，或显式选择 disk emergency mode；不得只输出“增加 RAM”。

## 10. 风险与缓解

### 10.1 Rank 间本地路径不同

风险：同层一个 rank direct permanent、另一个 rank cache fill，可能暴露隐藏的 raw expert
ID/slot ID 或 collective ordering假设。

缓解：增加同层 mixed residency的分布式 reference测试；在 execution adapter中显式记录
每 rank action，all-reduce保持无条件且顺序固定。

### 10.2 Whole-layer 粒度导致舍入损失

风险：aggregate byte总量可行，但完整 layer shard粒度使边界配置没有整数解。

缓解：frontier报告舍入后的最大 removable bytes和精确 shortfall。第一版不静默降到
per-expert permanent；若差额很小，先通过调整 cache/KV/reserve解决。

### 10.3 启动临时峰值

风险：最终 plan可行，但并行 loader同时 materialize多个 layers造成瞬时 host OOM。

缓解：保持 layer streaming；GPU permanent优先加载并在 copy完成后立即丢弃host pages；
对所有 ranks限制in-flight layers，并将 loader transient纳入host reserve。

### 10.4 显存碎片与 CUDA Graph

风险：静态byte proof通过，但allocation顺序或graph pool需要连续空间。

缓解：先分配 immutable permanent store，再分配dynamic/KV；记录allocated/reserved；
graph reserve使用实测上界，失败时报告对应项而不是启用disk。

### 10.5 安全预算过于保守

风险：`MemAvailable - reserve` 可能使理论总量看似不足。

缓解：默认继续保守；日志完整报告原始MemAvailable和扣减项。用户可以显式提高host
budget，但不能通过默认全量RAM分配重新引入OOM风险。

## 11. 文件级改动清单

| 文件 | 修改 |
|---|---|
| `python/freetoken/engine/moe_placement.py` | 删除 proportional split；增加 rank frontier、node组合、fixed KV语义和aggregate日志字段 |
| `python/freetoken/engine/engine.py` | 正确构造fixed/auto KV输入；消费rank-specific plan；校验实际pool；normal模式禁止disk source |
| `python/freetoken/engine/config.py` | 明确KV fixed target与auto minimum字段/注释 |
| `python/freetoken/moe/offload_cache.py` | 验证rank-local permanent与dynamic source共存；补充runtime disk计数 |
| `python/freetoken/moe/permanent_store.py` | 验证非对称rank plan所需allocation/staging契约 |
| `python/freetoken/models/deepseek_v4/moe.py` | 验证同层跨rank不同residency路径和collective一致性 |
| `python/freetoken/server/args.py` | 澄清KV与disk emergency CLI语义，禁止隐式disk fallback |
| `benchmarks/bench_decode_moe.py` | normal P3命令不带disk；记录plan checksum、fixed KV和runtime storage telemetry |
| `tests/engine/test_moe_placement.py` | 增加aggregate可行/local不可行及fixed KV regression |
| `tests/moe/test_gpu_only_residency.py` | 增加rank-specific layer tier和source释放测试 |
| `docs/moe-hybrid-execution-implementation-plan.md` | 修正P3状态与disk-backed定位，链接本计划 |

## 12. 完成定义

本修正的完成标志不是“模型能够在低 RAM 机器上返回 HTTP 200”，而是：planner在加载前
证明一个无 runtime disk的VRAM + RAM计划，loader精确落实该计划，runtime不读取专家
checkpoint，固定KV和cache没有被静默改写，且真实TP4 weighted服务在正确性、容量和性能
验收中全部通过。
