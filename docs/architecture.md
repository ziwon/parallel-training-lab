# Architecture

[Back to the README](../README.md)

The deterministic TypeScript engine computes one finite training step. React displays its records and chooses which snapshot to inspect. UI playback never schedules compute, communication, or storage.

## Engine boundary

```ts
const config = parseConfig(input);
const run = simulate(config);
const numerical = executeNumerics(run);
const frame = snapshotAtEvent(run, eventIndex);
```

`Simulation` contains the validated configuration, layers, allocations, scheduled operations, structured events, memory snapshots, and total modeled duration. Operation prerequisites refer to another operation's start or end. Allocations have stable IDs and describe virtual storage rather than browser-sized arrays.

## Module map

| Module                              | Responsibility                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/engine/types.ts`               | Configuration, operation, allocation, event, snapshot, and session contracts                |
| `src/engine/config.ts`              | Defaults and validation of external configuration                                           |
| `src/engine/workload.ts`            | Strategy identities, matrix coordinate ownership, pipeline stage and expert routing helpers |
| `src/engine/model.ts`               | Virtual layers, dtype constants, and shard divisibility                                     |
| `src/engine/ownership.ts`           | Replica, flat-shard, matrix, stage, expert ownership and sample partitions                  |
| `src/engine/strategies/`            | Five strategy operation graphs and explicit buffer lifetimes                                |
| `src/engine/cost.ts`                | Compute and collective duration formulas                                                    |
| `src/engine/scheduler.ts`           | Deterministic dependency and resource scheduling                                            |
| `src/engine/memory.ts`              | Residency ledger, consumer lifetime checks, category totals, and peaks                      |
| `src/engine/simulation.ts`          | Engine entry point, snapshot lookup, and operation status                                   |
| `src/engine/numerics.ts`            | Seeded fixture, independent reference, Adam, and event-driven numerical replay              |
| `src/engine/scenarios.ts`           | Fixed-problem comparison configurations                                                     |
| `src/engine/serialization.ts`       | Versioned share links and export/import boundaries                                          |
| `src/engine/nccl.ts`                | Verified NCCL Lab URL contract and size conversion                                          |
| `src/App.tsx` and `src/components/` | Configuration, playback, coordinated inspection, and visualization                          |

## Scheduling and replay

Strategy compilation records every operation and allocation before playback. The scheduler places operations after data and resource prerequisites. Each rank has one compute lane. All collectives and matched pipeline transfers use one ordered communication lane for the process group. Start edges permit bounded prefetch during computation. Completion edges prevent premature consumers.

Every start and completion becomes an ordered event. Equal-time events preserve causality, including one-rank collectives whose duration is zero. The memory ledger applies those events and emits snapshots. Numerical replay follows completion events and changes representative values only when the corresponding operation finishes.

The static graph was chosen because dense layers, GPipe microbatches, and the fixed MoE router produce a known finite execution graph. It keeps collective order, readiness explanations, and replay inspectable without a second set of mutable rank-program counters. Future conditional execution would need a new execution contract rather than extra UI state.

## Shared configuration and imported data

The URL contains a schema version and a complete session record. A session consists of the input configuration, comparison flag, and scenario ID. Unsupported versions, missing fields, invalid enums, unsupported batches, and oversized payloads fail at the boundary. Schema 2 requires the microbatch count. Complete schema 1 DDP/FSDP inputs migrate explicitly to schema 2 configuration; old numerical results remain archival. Comparison pairs are validated at this boundary too, preventing a valid TP-only batch from silently entering an incompatible DDP/FSDP comparison.

Export includes the inputs, explanatory trace, memory history, and final numerical results. Repeated numerical boundary snapshots are omitted and regenerated on import, bounding file growth for dense matrices and comparison runs. Import returns only a validated session. The application regenerates the graph, ledger, and representative arithmetic. It never evaluates imported code, accepts imported operation timings as authoritative, or fetches data from a trace-provided URL.

## Verification boundaries

The arithmetic tests include hand-checkable scalar and dense-matrix fixtures and finite differences. Dense strategies share one seeded four-matrix problem. EP has its own seeded eight-expert reference with the same fixed routes. TP reconstructs by explicit matrix indices; PP accumulates normalized microbatch contributions; EP checks token identity through all four exchanges. Strategy tests compare distributed gradients and updates with the full-batch reference. Scheduling tests inspect graph dependencies and lane overlap. Ledger tests inspect allocation identity, release times, category totals, and OOM history. Serialization tests validate round trips and operation-specific NCCL links. Browser tests cover the assembled interface.

The tests prove behavior of this educational model. They do not establish equivalence to PyTorch's allocation policy or measure GPU performance.

## Strategy-specific records

The shared operation record retains stable prerequisites, lane, participants, reads, allocations, and releases. `microbatch` identifies GPipe work. `phaseTag` distinguishes activation sums, input-gradient sums, stage-transfer direction, and each EP dispatch/combine phase. `reduction` distinguishes TP SUM from data-parallel averaging. Communication payload sizes are supplied by the strategy, rather than assumed to equal parameter size.

Memory and numerical replay consume the same scheduled boundary sequence. The representative parameter snapshot includes matrix indices for TP's noncontiguous column ownership. PP and EP ranks have empty parameter arrays for unowned layers or experts. UI inspection must represent this absence instead of treating every non-DDP rank as an FSDP shard.
