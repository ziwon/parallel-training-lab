# Model reference

[Back to the README](../README.md)

This reference describes simulator version 2.0.0. It is an educational execution model, not a PyTorch trace or a prediction of production scaling.

## Fixed training problem

The global batch stays fixed across a comparison. The data-parallel batch identity is:

```text
globalBatch = localMicrobatch × dataParallelRanks × accumulationSteps
accumulationSteps = 1
```

Supported rank counts are 1, 2, 4, and 8, except PP, which supports 1, 2, or 4 nonempty stages for the four-layer model. Global batch is an integer from 1 to 128. DDP/FSDP and EP source batches must divide evenly across ranks. PP uses 1, 2, 4, or 8 microbatches, and the global batch must divide evenly by that count. TP has one data replica: all tensor ranks cooperate on the same global samples. No configuration silently rounds, drops, or adds samples.

DDP/FSDP split sample IDs into equal contiguous batches and average their local mean gradients. PP accumulates contributions normalized by the global batch, so `globalBatch = microbatchSize × microbatches × 1 data replica`. EP initially partitions source tokens, then sends each token to one expert. Its loss uses the global token count, not the number routed to an individual expert. Every comparison preserves model workload, seed, optimizer, learning rate, global batch, and model scale. Dense and MoE runs are different learning problems and are never presented as a fixed-problem speedup comparison.

## Virtual layers and representative arithmetic

The virtual sequential model has four dense linear layers without biases. Its widths are 1024, 2048, 4096, 2048, and 1024. A positive integer model scale from 1 to 16 multiplies each layer's parameter count and saved-input size. Scale is an aggregate virtual workload multiplier, not a change to the tiny numerical fixture.

```text
layerElements = inputWidth × outputWidth × modelScale
savedInputBytes = localMicrobatch × inputWidth × 128 × modelScale × 4
```

The 128 positions per sample are a fixed virtual activation extent. At scale one the parameter groups contain 2,097,152, 8,388,608, 8,388,608, and 2,097,152 elements. Every group must contain a whole number of elements per rank. FSDP shards are contiguous half-open ranges, with exactly one owner for each element; TP column slices instead use explicit matrix coordinates. Each TP parameter and optimizer allocation exports a `matrixSlice` with axis, half-open coordinate bounds, matrix dimensions, and virtual copy count; its `range` is only the local storage extent, not a contiguous range of global matrix entries. The simulator rejects incompatible groups instead of padding them.

The browser computes a separate representative model with four dense 8×8 linear matrices, stored in input-major order. With row-vector notation, each layer evaluates `h[l+1] = h[l] W[l]`. This fixture has 256 actual weights regardless of virtual model scale, and its off-diagonal entries make matrix partitions observable. DDP, FSDP, TP, and PP use exactly the same dense fixture and initial data.

EP instead uses eight independent 8×8 linear experts, for 512 representative weights. Its virtual experts are each 1024×1024, with model scale and 128 virtual positions applied as above. Each sample is one representative token. The fixed top-1 router selects `expert = (sampleId × 5 + seed) % 8`; contiguous expert ranges are owned by ranks. There is no learned router, top-k duplication, capacity dropping, padding, or load-balancing loss.

Loss is the sum of squared prediction errors divided by twice the number of global samples and output channels. Numerical operations occur at completion events. The distributed executor computes partial products, communication results, gradients, and updates; it does not copy reference results into ranks. The full-batch reference and finite differences provide independent checks. Agreement uses an absolute tolerance of `1e-10`.

A scalar check uses `x=2`, `w=3`, target `1`: prediction `6`, half squared error `12.5`, and gradient `10`. A matrix check uses `X=[1,2]`, `Wc=[[1,2],[3,4]]`, `Wr=identity`, and target `[1,0]`: output `[7,10]`, loss `34`, output derivative `[3,5]`, `dWc=[[3,5],[6,10]]`, and `dWr=[[21,35],[30,50]]`. These calculations expose missing TP sums or incorrect batch normalization.

JavaScript numbers perform the arithmetic. FP32 labels describe four-byte storage in the virtual ledger. The application does not emulate FP32, FP16, or BF16 rounding.

## DDP

Each rank owns complete parameter and Adam-moment replicas. Local backward traverses layers in reverse order. A bucket contains one, two, or four layers in that same order. It can start AllReduce only when every participating rank has produced all gradients in the bucket. Collectives use the same bucket order on every rank. Later backward computation has no dependency on an unrelated bucket's AllReduce, so communication can overlap it.

With overlap disabled, bucket communication also waits for the full backward pass. Adam waits for all local backward work and all gradient reductions. Replicas begin with equal weights and use equal synchronized gradients.

These mechanisms follow the [PyTorch DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html). The simulator does not model DDP construction-time broadcast, dynamic bucket rebuilding, unused parameters, or autograd hooks.

## FSDP

Each rank owns a parameter shard and two moment shards for every layer. AllGather materializes a full temporary parameter buffer before local computation. Each rank still evaluates every layer on its own samples. Sharding model state does not divide an individual layer's mathematical computation across ranks as tensor parallelism does. The distinction follows NVIDIA's [parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

With `reshardAfterForward=true`, each rank releases its gathered buffer after its forward use. Backward gathers that layer again. With the setting false, the gathered buffer remains through backward, and the backward gather is omitted. Backward produces a full local gradient. ReduceScatter averages those gradients and writes each rank's owned output shard. Adam updates local parameter shards.

This lifecycle follows the [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and the [`fully_shard` API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html). The simulator applies one uniform policy to four layer groups. It has no root module with additional parameters or a special root retention policy. It does not reproduce `DTensor`, CUDA streams, caching allocators, runtime buffer reuse, or exact hook scheduling.

## Backward prefetch

Depth zero gathers a group's parameters only after the preceding backward group completes. Depth one schedules the next group's actual AllGather after all ranks start the current backward group. An additional dependency waits for the group two positions earlier to finish. This bounds gathered parameter residency to the current group plus one future group per rank.

The shared queue alternates according to the selected policy. With depth one, its backward order is `AG(3), AG(2), RS(3), AG(1), RS(2), AG(0), RS(1), RS(0)`. Each operation also observes its data prerequisites. The prefetched output allocates at AllGather start, including time when the current group is computing. A busy collective lane can delay prefetch. Retain mode needs no backward gather, so prefetch has no effect and its control is disabled.

The official [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) describes implicit and explicit prefetch. This simulator implements only the bounded backward policy above. It does not model forward prefetch or arbitrary prefetch depth.

## Tensor parallelism

For `Y = XW`, layers 1 and 3 partition weight columns. Each rank computes complete output features for its owned columns. Layers 2 and 4 partition weight rows and consume the corresponding input-feature slice. Their full-width outputs are partial sums. An AllReduce with **SUM** completes those outputs before the next layer or loss can consume them.

Backward through a row partition produces a local input-gradient slice. Backward through a column partition produces a full-width partial input derivative, requiring another AllReduce SUM. The first layer's input derivative is reduced too, providing a uniform trace. Parameter gradients belong to unique matrix partitions and already include all global samples, so they are not averaged across TP ranks. Adam updates only owned coordinates. Reconstruction uses explicit matrix indices because a flattened column partition is not contiguous.

This follows the column/row pairing described in the [PyTorch tensor-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html). The teaching model has no sequence parallelism, attention, bias handling, device mesh composition, or DTensor execution. Parameter partition storage is `layerElements / ranks`, but activation storage follows feature layout: column inputs are replicated and row inputs are sliced. Partial outputs, summed outputs, and input derivatives have explicit separate allocation lifetimes.

## Pipeline parallelism

PP assigns contiguous complete layers to stages and uses a **GPipe all-forward/all-backward** schedule. Every microbatch passes through all stages. A matched send/receive operation moves activations across each stage boundary. Backward sends input derivatives in the reverse direction. The receiver cannot read its destination buffer until the transfer completes.

All forward microbatches finish before backward begins, and backward visits microbatches in reverse order. Each microbatch saves its own inputs until matching backward use. A layer's gradient allocation is created once, accumulates microbatch contributions normalized by the global batch, and stays resident through Adam. One optimizer update follows the complete accumulation. One-rank PP has the same arithmetic and no stage transfers.

The named schedule is grounded in [PyTorch ScheduleGPipe](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe). There is no 1F1B, interleaved schedule, checkpoint recomputation, or universal pipeline-bubble formula. Idle gaps emerge from scheduled dependencies. All communication, including transfers on disjoint stage boundaries, shares the simulator's conservative single communication lane.

## Expert parallelism

Each rank owns complete experts, their gradients, and two Adam moment states. Source ranks first partition the global tokens contiguously. Four actual all-to-all events bracket the expert computation:

1. Dispatch token inputs from sources to their selected expert owners.
2. Combine expert outputs back into source-token order.
3. Dispatch output derivatives from sources to expert owners.
4. Combine input derivatives back to their sources after expert backward.

Expert forward and backward process only routed tokens. An expert receiving no tokens has zero parameter gradients. The loss denominator is always `globalBatch × outputWidth`, so a lightly used expert is not overweighted. Loss and output-derivative arithmetic is fused into the output-combine and backward-dispatch boundaries; no separate loss-kernel cost is modeled. The final input derivatives have no downstream trainable consumer in this bounded workload, but their return traffic is included.

The routing mechanism is grounded in [Megatron Core's MoE dispatcher documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html). This is a fixed-router, single-linear-expert teaching example, not a complete transformer MoE. It does not train the router, drop tokens, use top-k gates, overlap multiple MoE microbatches, or combine EP with DP/TP/PP. The EP comparison uses one versus multiple ranks on the same MoE workload.

## Storage ledger

Every allocation has an ID, rank, layer, category, ownership label, byte count, and element range. Events allocate or release IDs. Current memory is the sum of distinct resident allocations. Category peaks and the total peak are maxima over event snapshots. The sum of category peaks can exceed the total peak because categories can peak at different events.

| Allocation                  | Size per rank                   | Lifetime                                                   |
| --------------------------- | ------------------------------- | ---------------------------------------------------------- |
| DDP parameters              | `4 × layerElements`             | Entire run                                                 |
| FSDP parameters             | `4 × layerElements / ranks`     | Entire run                                                 |
| Adam moments                | Twice the owned parameter bytes | Entire run, initialized eagerly                            |
| Gathered parameters         | `4 × layerElements`             | Gather start through the last forward or backward consumer |
| DDP gradients               | `4 × layerElements`             | Backward start through Adam completion                     |
| FSDP full gradient inputs   | `4 × layerElements`             | Backward start through ReduceScatter completion            |
| FSDP output gradient shards | `4 × layerElements / ranks`     | ReduceScatter start through Adam completion                |
| Saved activations           | `savedInputBytes` above         | Forward start through the matching backward completion     |

TP and PP reserve separate saved-input copies and temporary output/derivative buffers. A partial TP tensor remains resident through its SUM, and the complete result stays until its consumer finishes. PP gradient accumulation allocates once per owned layer; microbatch-specific saved inputs release after backward. EP source inputs exist at initialization; dispatch reserves receive buffers. Expert forward saves a separate input copy, and the dispatch payload releases after the final local expert consumes it. Return buffers and outgoing payloads coexist during each exchange. Bottom input derivatives release after their final modeled return because no upstream trainable consumer exists.

DDP buckets alias gradient tensors and add no second allocation. FSDP uses separate gathered parameter and reduced-gradient output buffers. Full reduction inputs and output shards coexist during ReduceScatter. A destination buffer can be allocated while its contents are still unavailable. Compute waits for AllGather completion before reading it.

DDP/FSDP activations are local to a rank's samples, not sharded with FSDP model state. TP activation sizes follow the matrix layout; PP retains inputs per microbatch; EP retains routed expert inputs. Parameters, gradients, activations, and both moment states all use FP32 storage. There are no separate master weights or conversion buffers. Communication scratch is an explicit zero category. Loss outputs, targets, allocator caches, fragmentation, and kernel workspaces are excluded. DDP/FSDP transient input-gradient workspace is excluded; TP, PP, and EP communication-carrying output/input-gradient buffers are explicitly counted.

The ledger rejects duplicate allocation, release of an absent allocation, and release while another operation still consumes the allocation. Exceeding capacity sets OOM on the responsible snapshot and every later snapshot, even after current memory falls below capacity. It never clamps bytes or frees buffers to conceal the excess. Simulation continues so the offending lifetime remains inspectable.

## Adam update

The optimizer performs one step with `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. Both moments start at zero. There is no weight decay.

```text
m = 0.9 × 0 + 0.1 × gradient
v = 0.999 × 0 + 0.001 × gradient²
mHat = m / (1 - 0.9)
vHat = v / (1 - 0.999)
parameterNext = parameter - learningRate × mHat / (sqrt(vHat) + 1e-8)
```

DDP applies this update to complete replicas. FSDP applies it to each owned shard, TP to matrix coordinates, PP to stage layers, and EP to whole owned experts. Gradients are released after their optimizer consumer completes. Parameter and moment state remains resident.

## Dependencies, resources, and modeled time

Each rank has one compute lane. The process group has one serialized communication lane shared by AllGather, AllReduce, ReduceScatter, AllToAll, and matched pipeline send/receive operations. Compute can overlap communication, but two compute operations cannot occupy the same rank's compute lane and two communication operations cannot overlap each other. Every operation waits for its explicit prerequisites and its resource. The model assumes independent compute and communication resources and omits their real hardware interference.

Collective participants must all be ready. Rank-readiness values expose the delay caused by a late participant. Resource-readiness values expose the delay from preceding communication. No unrelated DDP layer imposes a global barrier. FSDP's next forward gather waits for all ranks to reach that group's gather request, so its collective rendezvous can synchronize local progress.

For `n` ranks and logical bytes `S`, the number of ring rounds is `2(n-1)` for AllReduce and `n-1` for AllGather or ReduceScatter.

```text
transferredBytesPerRank = S × rounds / n
collectiveDurationMs = rounds × latencyUs / 1000
                    + transferredBytesPerRank / (bandwidthGBps × 1,000,000)
```

`GB/s` uses decimal gigabytes. Memory labels use binary KiB, MiB, and GiB. `S` is the full per-rank buffer for AllReduce, total gathered output for AllGather, and total input before scattering for ReduceScatter. It is not the shard size. The transfer factors follow the [NCCL-tests performance definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md). They describe logical traffic, not physical routes.

A pipeline transfer uses `latencyUs / 1000 + payloadBytes / (bandwidthGBps × 1e6)`. Its logical size is the entire matched message, not a collective group size. EP builds source-to-owner token counts. For each exchange, `logicalBytes` is the maximum per-rank total outgoing payload including self-routed tokens. `transferredBytesPerRank` is the maximum across all ranks of outgoing or incoming remote payload, excluding self. Duration is `(ranks - 1) × latencyUs / 1000 + maxRemoteBytes / (bandwidthGBps × 1e6)`. This is a bounded synthetic exchange cost, not measured NCCL performance. One-rank EP has zero communication time and remote bytes.

Forward costs two FLOPs per owned virtual weight per processed sample-position. Backward costs four. DDP/FSDP use full-layer weights and local samples; TP uses owned matrix elements and the entire global batch; PP uses owned layers and microbatch samples; EP uses owned experts and their actual routed-token counts. Duration in milliseconds is FLOPs divided by `computeTFLOPS × 1e9`. Adam uses a fixed modeled 100 GB/s memory budget and 28 bytes of traffic per owned parameter element. The selected synthetic slow rank multiplies its compute durations, including Adam, by the configured factor. Communication duration itself stays unchanged, though participant readiness can delay it.

The scheduler uses start and completion dependency edges with deterministic ordering at equal timestamps. One-rank collectives have zero duration and zero transferred bytes. The uniform FSDP ledger still materializes separate full and reduced-output buffers at one rank. This conservative storage choice avoids an undocumented special-case alias and is not a claim about a real single-device FSDP allocation.

Playback speed, animation frames, wall-clock time, and browser visibility never enter the scheduler. Event stepping restores a snapshot by index. Scrubbing selects the last event at or before a simulated time. Comparison panels use the same simulated time, even if their event boundaries differ.

## NCCL Lab link contract

The integration targets [NCCL Lab](https://nccl-lab.pages.dev/) and its [inspected parser revision](https://github.com/ziwon/nccl-lab/blob/b912236e818d812e06ebdd0d33359eac9599aca1/src/simulation.ts). Links use supported fields only: `op`, `alg`, `n`, `size`, `root=0`, `red`, and an optional `bw`. Ring is used for AllReduce/AllGather/ReduceScatter, and Naive for AllToAll, matching the parser's supported algorithms.

`size = logicalBytes / 1,048,576` converts bytes to MiB. DDP AllReduce and FSDP ReduceScatter use `red=Average`; TP partial-result AllReduce uses `red=Sum`. AllGather and AllToAll do not reduce. An EP link uses its maximum per-rank input size as an explicitly illustrative uniform AllToAll example. NCCL Lab uses equal peer chunks, so it cannot reproduce this lab's unequal routing matrix or exact traffic. Pipeline point-to-point events have no NCCL Lab URL because its parser exposes no matched send/receive operation. Supported bandwidth presets are 16, 32, 50, and 100 GB/s. Other values are omitted and the inspector explains NCCL Lab's default 50 GB/s. Latency is not a supported query field and is not transferred. One-rank collectives have no external link because NCCL Lab accepts two or more ranks.

The tools both label storage FP32, but NCCL Lab's deterministic scalar samples illustrate contribution movement. They do not reproduce this fixture's training arithmetic. Only collective configuration is sent in the link. Imported traces, data samples, and parameters are never included.

## Serialization and exclusions

Share links contain schema version 2 and a complete validated session with its seed. JSON exports additionally contain simulator version, model assumptions, virtual layer metadata, numerical fixture metadata and inputs, operation records, events, allocations, memory history, and final numerical results. Repeated numerical boundary snapshots are regenerated rather than duplicated in the file, keeping maximum-rank comparisons within the import size limit. Import accepts at most 10 MiB, rejects unsupported versions and invalid sessions, and regenerates the simulation from inputs. Complete version 1 DDP/FSDP links and exports migrate by adding the default microbatch field. This migration preserves configuration, but regenerates version 2 dense arithmetic; the previous diagonal fixture and archived version 1 numerical results are not replayed. Saved event and result data is archival rather than trusted replay input.

Hybrid combinations of strategies, context parallelism, arbitrary pipeline schedules, learned MoE routing, and data-parallel accumulation beyond one are outside this release. So are activation checkpointing, CPU offload, mixed precision, elastic ranks, real profiler import, physical topology, collective algorithm selection, CPU launch overhead, protocols, channels, and network contention beyond the single communication lane.
