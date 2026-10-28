# Scenario guide

[Back to the README](../README.md)

Every comparison holds the model, seed, global batch, optimizer, and learning rate fixed. Changing bandwidth, retention, prefetch, or synthetic rank speed changes the execution model rather than the learning problem. Both panels follow the same simulated clock. Their inspectors can select different event boundaries.

## DDP versus FSDP

The left run uses DDP and the right run uses FSDP. At initialization, compare persistent parameter and optimizer allocations. DDP has complete replicas. FSDP has one owned shard of every group per rank.

Step into forward. FSDP must gather a group's parameters before its local computation. Its temporary memory rises during that gather. DDP can use its resident replica immediately. At backward, compare DDP's AllReduce buckets with FSDP's ReduceScatter operations. Both average gradients over the same samples.

Each FSDP rank still computes the whole local layer. The shard ownership display explains state storage, not tensor-parallel work partitioning. The numerical result should agree with the full-batch reference and the other strategy within the displayed tolerance.

## Reshard versus retain

Both runs use FSDP with backward prefetch off. The left run releases gathered parameters after forward. The right run retains them until backward finishes using them.

Inspect the memory ledger after a forward group completes. Reshard releases that rank's full gathered buffer. Retain keeps it resident. During backward, only the reshard run needs another AllGather for each group. The comparison exposes the memory cost of retaining buffers and the communication saved by doing so.

Retain mode disables backward prefetch because the complete parameter buffers are already present. It does not create a hidden second gather.

## Prefetch tradeoff

Both runs use FSDP with resharding. The left run uses depth zero and the right run uses depth one. Inspect a backward group's compute operation together with the next group's AllGather.

With depth one, the next gather can run while the current group computes. The output buffer exists from gather start and contributes to memory before its own backward use. The next group still waits for the gather to finish. The shared collective lane can delay prefetch behind earlier communication.

A shorter modeled step is not guaranteed for every cost configuration. Even when overlap is possible, extra gathered-buffer residency is real. Compare the event-linked memory history rather than assuming peak memory equals persistent state divided by rank count.

## One slow rank

The left run has no synthetic slow rank. The right run doubles compute durations on the last rank while preserving all numerical inputs. Select a collective and compare each participant's readiness time. Faster ranks can reach a collective before the slow rank supplies its gradients or gather request.

The wait comes from a prerequisite or occupied resource. It is not an unexplained global barrier added after every layer. With one rank, this scenario still slows local computation, but there are no other ranks waiting and communication remains a no-op.

## FSDP versus TP

Both runs use the same dense matrices, seed, and global batch. FSDP splits samples among ranks and materializes complete layer parameters. TP cooperates on the same samples using matrix partitions. Inspect TP's row-layer forward AllReduce: it sums partial activations, whereas DDP averages local parameter gradients. Backward column-layer sums similarly complete the input derivative. FSDP's parameter shards and TP's computational slices have different meanings even when persistent bytes are similar.

## GPipe microbatches

The left run uses one microbatch; the right uses the configured microbatch count. Both process the exact same global batch through identical stage layers. With more microbatches, stages can compute different microbatches concurrently when transfer prerequisites and the shared communication resource permit it. All forward work finishes before the backward phase.

Select a stage transfer to follow the producer, reserved receive allocation, and dependent compute. Inspect saved activations for each microbatch and the layer gradient that accumulates across them. The single Adam update must match the dense reference regardless of microbatch count. The four-layer model supports up to four nonempty stages.

## MoE distribution

The left run places all eight experts on one rank; the right distributes them across the configured rank count. The router, token IDs, expert parameters, optimizer, and global token count are identical. This comparison uses a separate MoE workload and does not imply a dense-versus-MoE performance result.

Follow forward dispatch, expert compute, output combine, derivative dispatch, expert backward, and input-gradient combine. An expert owns complete weights and receives only selected tokens. Exactly one expert handles each token. The loss uses the global token denominator, including tokens handled by other experts. NCCL Lab links show a clearly labeled uniform AllToAll illustration because the source/destination token counts need not be equal.

## Capacity and one-rank checks

Lower memory capacity to see the first event that exceeds it. OOM does not clamp the plot or remove allocations. A completed simulation after an OOM marker is an explanatory trace, not a feasible hardware execution.

Select one rank to inspect zero-time collective boundaries and unchanged numerical semantics. FSDP keeps its explicit out-of-place buffer policy at one rank, so separate temporary storage can remain visible. The inspector explains why no one-rank link opens in NCCL Lab.
