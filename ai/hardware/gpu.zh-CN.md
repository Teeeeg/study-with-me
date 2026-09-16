---
title: GPU 性能、通信与调度
description: 从性能指标、NVLink 与 PCIe 通信，到集群中的拓扑感知调度。
lang: zh
ref: gpu
nav_order: 1
math: true
---

# GPU 性能、通信与调度

GPU 集群不能只看“有多少张卡”。单卡是否真正发挥算力、多卡之间走哪条通信路径、
调度器分配的 GPU 是否处在合适的拓扑中，都会直接影响训练吞吐。分析时可以按三层逐步
定位：**计算是否饱和 → 通信是否受限 → 调度是否破坏了理想拓扑**。

## GPU 的性能指标

### GPU 的运行性能

`nvidia-smi` 中的 `GPU-Util` 适合回答“采样周期内 GPU 是否有工作”，不适合回答
“GPU 的计算能力用了多少”。它统计采样周期内一个或多个 kernel 在 GPU 上执行的时间
占比；即使 kernel 只使用少量执行单元，或者主要在搬运数据，这个值也可能接近 100%。

判断运行性能时，需要组合观察不同层级的指标：

![GPU performance signals from activity to workload outcomes](../../.asset/gpu/gpu-performance-signals.drawio.svg)

| 指标 | 回答的问题 | 主要局限 |
| --- | --- | --- |
| `GPU-Util` | GPU 在采样周期内是否在执行 kernel | 不表示用了多少 SM、Tensor Core 或 FLOPS |
| SM Active / SM Efficiency | SM 在多少周期内有活跃 warp | SM 活跃不等于所有执行单元都被充分利用 |
| SM Occupancy | SM 可容纳的活跃 warp 比例 | 高 occupancy 不保证 kernel 更快，也不等于计算吞吐高 |
| DRAM Active / Memory Utilization | 显存控制器或显存带宽有多忙 | 高值只说明数据搬运繁忙，不说明计算有效 |
| Tensor Core 利用率 | 矩阵计算单元是否被使用 | 只适用于能映射到 Tensor Core 的运算 |
| MFU（Model FLOPs Utilization） | 模型有效 FLOPS 占硬件理论峰值的比例 | 依赖模型 FLOPS 估算、精度与硬件峰值，不能直接逐 kernel 采样 |
| 吞吐与延迟 | 每秒处理多少 token/sample，或一次请求耗时多久 | 是最终结果指标，不能单独说明瓶颈位置 |

MFU 的常见定义为：

$$
\mathrm{MFU} =
\frac{\text{模型每步理论 FLOPs} \times \text{每秒完成的训练步数}}
{\text{GPU 数量} \times \text{单卡对应精度的峰值 FLOPS}}
$$

MFU 比 `GPU-Util` 更接近训练效率，但不同资料对模型 FLOPS、稀疏计算以及重计算的
计法可能不同，因此只应在口径一致时横向比较。

#### 用指标组合判断瓶颈

| 现象 | 可能的状态 | 下一步 |
| --- | --- | --- |
| `GPU-Util` 高，SM Active 高，吞吐稳定 | GPU 正在持续计算 | 再看 Tensor Core、指令吞吐和 kernel 效率 |
| `GPU-Util` 高，SM Active 低，DRAM Active 高 | memory-bound 或频繁搬运 | 检查算术强度、kernel fusion、数据布局和 H2D 拷贝 |
| SM Active 高，但 MFU 低 | kernel 持续运行但有效模型计算少 | 检查小 kernel、非 Tensor Core 算子、通信 kernel 和同步等待 |
| GPU 指标呈周期性低谷 | CPU/DataLoader 或通信供给不足 | 对照 CPU、磁盘、网络和 NCCL timeline |
| 功耗或时钟到达上限，性能不再增长 | 功耗墙、温度墙或频率限制 | 检查 clocks、power limit、temperature 和 throttling reason |

快速观察可使用：

```bash
# 设备级利用率、显存带宽、功耗和时钟的连续视图
nvidia-smi dmon -s pucm -d 2

# 查看温度、功耗和时钟限制
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,clocks.sm \
	--format=csv
```

这里的 `dmon` 与 `GPU-Util` 仍然主要用于快速筛查。需要准确区分计算、显存和 stall
原因时，应使用 DCGM Profiling Metrics、Nsight Systems、Nsight Compute 或框架
profiler。优化顺序通常是先以吞吐/MFU 确认存在差距，再用 timeline 找等待区间，最后
下钻到具体 kernel；不要只围绕一个利用率数字调参。

## GPU 间的通信

GPU 间通信的有效性能由**物理链路、拓扑距离、消息大小和集合通信算法**共同决定。
NCCL 会读取系统拓扑并为 AllReduce、AllGather、ReduceScatter 等操作选择通信路径，
但自动选择不能修复错误的布线、被禁用的 P2P 或不合理的 GPU 分配。

先用下面的命令了解机器，而不要根据 GPU 编号猜测拓扑：

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
```

`topo -m` 中常见的 GPU 间标识包括 `NV#`（经过若干条 NVLink）、`PIX`（同一 PCIe
switch）、`PXB`（经过多个 PCIe bridge）、`NODE`（同一 NUMA node）和 `SYS`
（跨 NUMA/socket）。具体标识随驱动和硬件变化，应以当前机器输出为准。

### NVLink

NVLink 是 GPU 之间的高速点对点互连。NVSwitch 则把多张 GPU 接入交换结构，使一个
NVSwitch 域内的 GPU 可以获得高带宽、低延迟的互连。它们适合张量并行（TP）这类
通信频繁、对延迟和带宽都敏感的工作负载。

需要区分三个概念：

- **NVLink 数量和代际**决定单条链路及聚合链路的理论峰值；
- **是否在同一 NVSwitch 域**决定任意 GPU pair 能否走等价的高速路径；
- **NCCL 实测带宽**是算法、消息大小和链路共同作用后的结果，不能直接等同于厂商标注
	的链路峰值。

同一 NVSwitch 域内，GPU 即使挂在不同 CPU NUMA node 下，GPU-to-GPU 数据仍可能完全
经 NVSwitch 传输，因此 NUMA 对这条路径影响很小。不过 CPU-to-GPU 和 NIC-to-GPU 的
路径仍受 NUMA/PCIe 亲和性影响，不能据此忽略所有 NUMA 配置。

### PCIe

PCIe 是 GPU 与 CPU、NIC 及其他设备之间的通用互连，也可在平台支持时承载 GPU P2P。
与 NVLink 相比，它通常带宽更低、拓扑层级更多，而且可能经过 PCIe switch、Host
Bridge 或 CPU socket 间链路。

常见路径从优到劣大致为：

![Direct NVLink communication compared with host-staged PCIe fallback](../../.asset/gpu/gpu-interconnect-paths.drawio.svg)

```text
同域 NVLink/NVSwitch
	→ 同一 PCIe switch 的 GPU P2P
	→ 同一 NUMA node 内经 Host Bridge
	→ 跨 CPU socket
	→ GPU → CPU 内存 → GPU 的 P2P fallback
```

是否能够使用 PCIe P2P 取决于 GPU、主板拓扑、IOMMU/ACS 和驱动配置。P2P 不可用或被
`NCCL_P2P_DISABLE=1` 禁用时，数据可能需要经 CPU 内存中转，产生两次 PCIe 传输和额外
拷贝。跨节点的 GPUDirect RDMA 则允许 NIC 直接访问 GPU 显存，减少 CPU 内存中转；此时
应优先让 GPU 使用同一 PCIe switch 或同一 NUMA node 下的 NIC。

### 如何测量通信性能

通信测试必须固定 GPU 组合、消息大小、集合操作和 NCCL 配置。`nccl-tests` 的
`all_reduce_perf` 通常同时给出：

- `algbw`：从数据量和操作耗时计算出的算法带宽；
- `busbw`：按集合通信的实际传输量折算的总线带宽，适合比较硬件路径利用程度。

对于 $N$ 个 rank 的 Ring AllReduce，NCCL 使用的折算关系为：

$$
\mathrm{busbw} = \mathrm{algbw} \times \frac{2(N-1)}{N}
$$

小消息通常由启动延迟主导，大消息才逐渐逼近链路带宽。因此比较两种拓扑时不能只测
一个很小的张量。建议先建立同 NVLink 域的基线，再测试跨域或跨 NUMA GPU pair，并用
`NCCL_DEBUG=INFO` 确认 NCCL 实际选择了 P2P、SHM、IB 还是 Socket 路径。

## GPU 调度

### 调度问题

Kubernetes 默认把 `nvidia.com/gpu` 当作不可分割的标量扩展资源。它能判断某节点是否
还有 4 张 GPU，却不知道这 4 张 GPU 之间是 NVLink 还是 PCIe，也不知道一个分布式作业
的所有 worker 是否必须同时启动。由此产生几个典型问题：

1. **资源碎片化**：单卡任务分散占用 GPU 后，剩余数量虽然足够，却无法组成满足 TP
	 通信要求的 GPU 组。重点不是编号是否连续，而是剩余 GPU 是否位于同一高速互连域。
2. **拓扑不感知**：调度成功不等于性能合理。跨 NVSwitch 域、跨 socket 或绑定远端 NIC
	 都可能让通信回退到较慢路径。
3. **缺少 Gang Scheduling**：分布式训练要求全部 rank 就绪。若只调度部分 worker，
	 已启动的 worker 会占着 GPU 等待 NCCL 初始化，形成资源空转。
4. **训练与推理目标冲突**：单卡推理偏好提高装箱率，多卡训练偏好保留完整的高速互连
	 域。两类负载混部时，需要配额、优先级或专用节点池来控制碎片。

对应的治理手段包括拓扑感知 Filter/Score、Volcano 等调度器的 coscheduling、队列与
优先级、低优先级任务抢占/重调度，以及按 NVSwitch 域划分节点池。Gang Scheduling
解决的是“所有 Pod 能否一起运行”，拓扑感知解决的是“它们运行在哪里”；两者不能互相
替代。

### 拓扑感知

拓扑感知调度可分为三层：

![Topology-aware scheduling from workload intent to GPU placement](../../.asset/gpu/gpu-topology-scheduling.drawio.svg)

| 层级 | 调度器需要保证什么 | 适用场景 | 常见策略 |
| --- | --- | --- | --- |
| GPU 互连 | TP 组优先位于同一 NVLink/NVSwitch 域 | TP、频繁集合通信 | 强约束 Filter，加拓扑 Score |
| CPU/内存 NUMA | CPU、内存、GPU 和 NIC 尽量位于同一 NUMA node | DataLoader、H2D、GPUDirect RDMA | Kubelet Topology Manager |
| 节点/网络 | TP 组不跨节点，DP 组使用合适的 IB/RoCE 网络 | 多机混合并行 | PodGroup、节点亲和性和 rank 映射 |

三种并行方式对拓扑的敏感度不同：

- **张量并行（TP）**每层都可能通信，应尽量把一个 TP 组放在同一节点、同一 NVSwitch
	域；
- **流水线并行（PP）**主要在 stage 边界通信，可在带宽和显存容量之间权衡；
- **数据并行（DP）**本来就常跨节点做梯度同步，重点是 IB/RoCE 带宽、NIC 亲和性和
	通信与反向计算重叠。

Kubelet 的 Topology Manager 可以协调 CPU Manager、Memory Manager 和 Device Plugin
给出的 NUMA hint。策略强度应按负载选择：普通推理可用 `best-effort`，对 H2D 或 RDMA
敏感的训练可考虑 `restricted`；`single-numa-node` 最严格，但也更容易因局部资源不足
而使 Pod Pending。GPU-to-GPU 已经通过单一 NVSwitch 域全互联时，不应为了 NUMA 对齐
无谓牺牲可调度性。

调度策略最终必须用实际工作负载验证：记录分配到的 GPU UUID，保存 `nvidia-smi topo
-m`，运行对应 GPU 组合的 NCCL 基准，并比较端到端 tokens/s 或 samples/s。只有同时
满足**能调度、通信路径正确、整体吞吐提升**，拓扑策略才真正有效。

## 参考资料

- [AI 集群运维与通信](https://github.com/ForceInjection/AI-fundamentals/tree/main/03_ai_cluster_ops)
- [NVIDIA DCGM Profiling Metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)