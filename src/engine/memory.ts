import { categories } from "./types";
import type {
  Allocation,
  Config,
  MemoryTotals,
  Operation,
  SimulationEvent,
  Snapshot,
} from "./types";

const totals = (): MemoryTotals => ({
  parameters: 0,
  gradients: 0,
  optimizer: 0,
  masterParameters: 0,
  activations: 0,
  temporary: 0,
  scratch: 0,
});

export function replayMemory(
  config: Config,
  allocations: Allocation[],
  operations: Operation[],
  events: SimulationEvent[],
): Snapshot[] {
  const storage = new Map(
    allocations.map((allocation) => [allocation.id, allocation]),
  );
  if (storage.size !== allocations.length)
    throw new Error("Allocation IDs must be unique.");
  for (const allocation of allocations) {
    if (
      !Number.isSafeInteger(allocation.bytes) ||
      allocation.bytes < 0 ||
      allocation.rank < 0 ||
      allocation.rank >= config.ranks
    )
      throw new Error(`Invalid allocation ${allocation.id}.`);
  }
  const byId = new Map(
    operations.map((operation) => [operation.id, operation]),
  );
  const resident = new Set<string>();
  const allocated = new Set<string>();
  const ready = new Set<string>();
  const executing = new Set<string>();
  const started = new Set<string>();
  const completed = new Set<string>();
  const locks = new Map<string, Set<string>>();
  const peaks = Array.from({ length: config.ranks }, totals);
  const peakTotals = Array.from({ length: config.ranks }, () => 0);
  const snapshots: Snapshot[] = [];
  let priorTime = 0;
  for (const event of events) {
    if (event.time < priorTime)
      throw new Error("Events must be ordered by simulated time.");
    priorTime = event.time;
    const operation =
      event.operation === null ? undefined : byId.get(event.operation);
    if (event.operation !== null && !operation)
      throw new Error(`Unknown operation ${event.operation}.`);
    if (event.phase === "start" && operation) {
      if (started.has(operation.id))
        throw new Error(`Operation started twice: ${operation.id}.`);
      for (const dependency of operation.prerequisites) {
        if (
          !(dependency.phase === "start" ? started : completed).has(
            dependency.operation,
          )
        )
          throw new Error(
            `Operation ${operation.id} started before prerequisite ${dependency.operation}.`,
          );
      }
      for (const id of operation.reads) {
        if (!resident.has(id))
          throw new Error(
            `Operation ${operation.id} reads absent buffer ${id}.`,
          );
        if (!ready.has(id))
          throw new Error(
            `Operation ${operation.id} reads unready buffer ${id}.`,
          );
        const consumers = locks.get(id) ?? new Set<string>();
        consumers.add(operation.id);
        locks.set(id, consumers);
      }
      started.add(operation.id);
      executing.add(operation.id);
    }
    for (const id of event.allocated) {
      if (!storage.has(id))
        throw new Error(`Missing allocation metadata for ${id}.`);
      if (allocated.has(id)) throw new Error(`Buffer allocated twice: ${id}.`);
      allocated.add(id);
      resident.add(id);
      if (event.phase === "initialize") ready.add(id);
    }
    if (event.phase === "end" && operation) {
      if (!executing.has(operation.id))
        throw new Error(`Operation ended without starting: ${operation.id}.`);
      for (const id of operation.reads) locks.get(id)?.delete(operation.id);
      for (const id of operation.allocate) ready.add(id);
      executing.delete(operation.id);
      completed.add(operation.id);
    }
    for (const id of event.released) {
      if (!resident.has(id)) throw new Error(`Release of absent buffer ${id}.`);
      if (locks.get(id)?.size)
        throw new Error(`Buffer ${id} released while a consumer is executing.`);
      resident.delete(id);
      ready.delete(id);
    }
    const current = Array.from({ length: config.ranks }, totals);
    for (const id of resident) {
      const allocation = storage.get(id)!;
      current[allocation.rank][allocation.category] += allocation.bytes;
    }
    const memory = current.map((rankTotals, rank) => {
      for (const category of categories)
        peaks[rank][category] = Math.max(
          peaks[rank][category],
          rankTotals[category],
        );
      const total = categories.reduce(
        (sum, category) => sum + rankTotals[category],
        0,
      );
      peakTotals[rank] = Math.max(peakTotals[rank], total);
      return {
        rank,
        current: rankTotals,
        peak: { ...peaks[rank] },
        total,
        peakTotal: peakTotals[rank],
        oom: peakTotals[rank] > config.capacityMiB * 1048576,
      };
    });
    snapshots.push({
      eventIndex: event.index,
      time: event.time,
      resident: [...resident],
      memory,
      completed: [...completed],
      executing: [...executing],
    });
  }
  if (executing.size)
    throw new Error("Event log ends with operations still executing.");
  return snapshots;
}
