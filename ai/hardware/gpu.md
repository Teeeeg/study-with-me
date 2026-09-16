---
title: GPU Performance, Communication, and Scheduling
description: GPU performance metrics, NVLink and PCIe communication, and topology-aware cluster scheduling.
lang: en
ref: gpu
nav_order: 1
math: true
---

# GPU Performance, Communication, and Scheduling

A GPU cluster is more than a count of available accelerators. Training
throughput depends on whether each GPU uses its compute resources effectively,
which path data takes between GPUs, and whether the scheduler preserves the
ideal hardware topology. A useful diagnostic order is: **check compute
saturation, locate communication limits, then verify placement topology**.

## GPU Performance Metrics

### Runtime Performance

`GPU-Util` in `nvidia-smi` answers whether the GPU executed one or more kernels
during a sampling interval. It does not answer how much of the GPU's compute
capacity those kernels used. A kernel that occupies only a few execution units
or mostly moves data can still produce a value close to 100%.

Performance analysis therefore requires metrics from several layers:

![GPU performance signals from activity to workload outcomes](../../.asset/gpu/gpu-performance-signals.drawio.svg)

| Metric | Question it answers | Main limitation |
| --- | --- | --- |
| `GPU-Util` | Was a kernel executing during the sample window? | Does not show how many SMs, Tensor Cores, or FLOPS were used |
| SM Active / SM Efficiency | During how many cycles did an SM have an active warp? | An active SM does not imply full use of its execution units |
| SM Occupancy | What fraction of the SM's possible active warps was resident? | High occupancy does not guarantee a faster kernel or high compute throughput |
| DRAM Active / Memory Utilization | How busy were the memory controllers or memory bandwidth? | High activity shows data movement, not useful computation |
| Tensor Core utilization | Were matrix execution units being used? | Applies only to operations that can map to Tensor Cores |
| MFU (Model FLOPs Utilization) | What fraction of theoretical peak FLOPS performed useful model work? | Depends on the model FLOPS estimate, numerical precision, and hardware peak |
| Throughput and latency | How many tokens or samples are processed per second, or how long does a request take? | Shows the outcome but does not identify the bottleneck |

A common definition of MFU is:

$$
\mathrm{MFU} =
\frac{\text{theoretical model FLOPs per step} \times \text{steps per second}}
{\text{number of GPUs} \times \text{peak FLOPS per GPU at the selected precision}}
$$

MFU is closer to training efficiency than `GPU-Util`, but sources may count
model FLOPS, sparsity, and recomputation differently. Compare MFU values only
when they use the same methodology.

#### Combining Metrics to Identify Bottlenecks

| Observation | Likely state | Next step |
| --- | --- | --- |
| High `GPU-Util`, high SM Active, stable throughput | The GPU is computing continuously | Inspect Tensor Core use, instruction throughput, and kernel efficiency |
| High `GPU-Util`, low SM Active, high DRAM Active | Memory-bound execution or frequent transfers | Check arithmetic intensity, kernel fusion, data layout, and H2D copies |
| High SM Active but low MFU | Kernels run continuously but perform little useful model work | Look for small kernels, non-Tensor-Core operators, communication kernels, and synchronization |
| Periodic troughs in GPU activity | The CPU, DataLoader, or communication layer is not supplying work | Correlate with CPU, storage, network, and NCCL timelines |
| Power or clock limits are reached without more performance | Power, thermal, or frequency throttling | Inspect clocks, power limit, temperature, and throttling reasons |

Use these commands for a quick first look:

```bash
# Continuous device-level utilization, memory, power, and clock view
nvidia-smi dmon -s pucm -d 2

# Inspect temperature, power, and SM clocks
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,clocks.sm \
	--format=csv
```

`dmon` and `GPU-Util` remain screening tools. Use DCGM Profiling Metrics,
Nsight Systems, Nsight Compute, or a framework profiler to distinguish compute,
memory, and stall causes accurately. A practical workflow first confirms a gap
with throughput or MFU, uses a timeline to find waiting intervals, and only then
drills into individual kernels. Do not optimize around one utilization number.

## Communication Between GPUs

Effective GPU communication performance depends on the **physical link,
topological distance, message size, and collective algorithm**. NCCL reads the
system topology and selects paths for AllReduce, AllGather, ReduceScatter, and
other collectives. Automatic selection cannot repair incorrect wiring, disabled
P2P, or a poor GPU allocation.

Inspect the machine instead of inferring topology from GPU indices:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
```

Common GPU-to-GPU labels in `topo -m` include `NV#` for NVLink paths, `PIX` for
devices under one PCIe switch, `PXB` for paths through multiple PCIe bridges,
`NODE` for paths within one NUMA node, and `SYS` for paths across NUMA nodes or
CPU sockets. Labels vary with hardware and driver versions, so use the output
from the actual machine.

### NVLink

NVLink is a high-speed point-to-point GPU interconnect. NVSwitch connects
multiple GPUs through a switching fabric, allowing GPUs in one NVSwitch domain
to use high-bandwidth, low-latency paths. These links are particularly valuable
for tensor parallelism (TP), which communicates frequently and is sensitive to
both latency and bandwidth.

Keep three concepts separate:

- **NVLink generation and link count** determine the theoretical per-link and
  aggregate peak bandwidth.
- **NVSwitch domain membership** determines whether GPU pairs have equivalent
  high-speed paths.
- **Measured NCCL bandwidth** includes the effects of the algorithm, message
  size, and physical path; it is not the same as the advertised link peak.

Within one NVSwitch domain, GPU-to-GPU traffic may stay entirely on NVSwitch
even when the GPUs are attached to different CPU NUMA nodes. NUMA then has
little effect on that particular path. CPU-to-GPU and NIC-to-GPU paths remain
sensitive to NUMA and PCIe affinity, so this does not make all NUMA placement
irrelevant.

### PCIe

PCIe is the general-purpose interconnect between GPUs, CPUs, NICs, and other
devices. It can also carry GPU P2P traffic when the platform supports it. PCIe
usually offers less bandwidth than NVLink and may traverse PCIe switches, host
bridges, or inter-socket links.

A typical path hierarchy from best to worst is:

![Direct NVLink communication compared with host-staged PCIe fallback](../../.asset/gpu/gpu-interconnect-paths.drawio.svg)

```text
NVLink/NVSwitch within one fabric domain
	→ GPU P2P under one PCIe switch
	→ Host Bridge within one NUMA node
	→ Cross-socket path
	→ GPU → CPU memory → GPU P2P fallback
```

PCIe P2P availability depends on the GPUs, motherboard topology, IOMMU/ACS, and
driver configuration. If P2P is unavailable or disabled with
`NCCL_P2P_DISABLE=1`, data may be staged through CPU memory, adding two PCIe
transfers and an extra copy. Across nodes, GPUDirect RDMA allows the NIC to
access GPU memory directly. In that case, prefer a NIC under the same PCIe
switch or NUMA node as the GPU.

### Measuring Communication Performance

A communication benchmark must fix the GPU pair or group, message size,
collective, and NCCL configuration. `all_reduce_perf` from `nccl-tests` usually
reports two bandwidth values:

- `algbw`: algorithm bandwidth calculated from payload size and operation time;
- `busbw`: bandwidth normalized by the data movement required by the
  collective, useful for estimating physical path utilization.

For Ring AllReduce with $N$ ranks, NCCL uses this conversion:

$$
\mathrm{busbw} = \mathrm{algbw} \times \frac{2(N-1)}{N}
$$

Startup latency dominates small messages; large messages gradually approach
the link's bandwidth ceiling. Do not compare topologies with only one small
tensor. Establish a baseline inside one NVLink domain, then test cross-domain
or cross-NUMA GPU groups. Use `NCCL_DEBUG=INFO` to confirm whether NCCL selected
P2P, SHM, IB, or Socket transport.

## GPU Scheduling

### Scheduling Problems

The default Kubernetes scheduler treats `nvidia.com/gpu` as an indivisible
scalar extended resource. It can determine that a node has four free GPUs, but
not whether those GPUs communicate over NVLink or PCIe, nor whether every
worker of a distributed job must start together. This creates several common
problems:

1. **Resource fragmentation:** single-GPU jobs can leave enough free GPUs in
   total but no suitable group for TP. Consecutive indices do not matter; a
   shared high-speed interconnect domain does.
2. **Topology-unaware placement:** a schedulable allocation is not necessarily
   a performant one. Crossing NVSwitch domains or CPU sockets, or binding a
   remote NIC, can force communication onto a slower path.
3. **Missing Gang Scheduling:** distributed training needs every rank online.
   If only some workers start, they reserve GPUs while waiting for NCCL
   initialization.
4. **Conflicting training and inference goals:** single-GPU inference favors
   tight packing, while multi-GPU training benefits from preserving complete
   high-speed fabric domains. Mixed clusters need quotas, priorities, or
   dedicated node pools to control fragmentation.

Mitigations include topology-aware Filter and Score plugins, coscheduling with
systems such as Volcano, queues and priorities, preemption or rescheduling of
low-priority jobs, and node pools aligned with NVSwitch domains. Gang
Scheduling answers whether all Pods can run together; topology-aware scheduling
answers where they should run. Neither replaces the other.

### Topology Awareness

Topology-aware scheduling operates at three levels:

![Topology-aware scheduling from workload intent to GPU placement](../../.asset/gpu/gpu-topology-scheduling.drawio.svg)

| Level | What the scheduler must preserve | Typical workload | Common policy |
| --- | --- | --- | --- |
| GPU fabric | Place a TP group in one NVLink/NVSwitch domain | TP and frequent collectives | Hard Filter constraint plus topology Score |
| CPU/memory NUMA | Keep CPU, memory, GPU, and NIC near one another | DataLoader, H2D, and GPUDirect RDMA traffic | Kubelet Topology Manager |
| Node/network | Keep TP within a node and map DP onto suitable IB/RoCE links | Multi-node hybrid parallelism | PodGroup, node affinity, and rank mapping |

Parallelism strategies differ in topology sensitivity:

- **Tensor parallelism (TP)** may communicate at every layer. Keep a TP group
  within one node and one NVSwitch domain whenever possible.
- **Pipeline parallelism (PP)** communicates mainly at stage boundaries and can
  trade interconnect performance against memory capacity.
- **Data parallelism (DP)** commonly synchronizes gradients across nodes. Its
  priorities are IB/RoCE bandwidth, NIC affinity, and overlap between
  communication and backward computation.

Kubelet's Topology Manager coordinates NUMA hints from CPU Manager, Memory
Manager, and Device Plugins. Choose policy strength for the workload. Ordinary
inference can use `best-effort`; training sensitive to H2D or RDMA traffic may
use `restricted`. `single-numa-node` is the strictest policy but can leave a Pod
pending when one NUMA node lacks enough local resources. If a single NVSwitch
domain already fully connects the GPUs, do not sacrifice schedulability merely
to align GPU-to-GPU traffic with CPU NUMA boundaries.

Validate every scheduling policy with the real workload. Record allocated GPU
UUIDs, save `nvidia-smi topo -m`, benchmark the assigned GPU group with NCCL,
and compare end-to-end tokens/s or samples/s. A topology policy succeeds only
when the job is schedulable, the communication path is correct, and total
throughput improves.

## References

- [AI Cluster Operations and Communication](https://github.com/ForceInjection/AI-fundamentals/tree/main/03_ai_cluster_ops)
- [NVIDIA DCGM Profiling Metrics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics)
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)