import { parseConfig } from "./config";
import { replayMemory } from "./memory";
import { modelFor, validateLayerSharding } from "./model";
import { schedule } from "./scheduler";
import { ddpPlan } from "./strategies/ddp";
import { fsdpPlan } from "./strategies/fsdp";
import { tpPlan } from "./strategies/tp";
import { ppPlan } from "./strategies/pp";
import { epPlan } from "./strategies/ep";
import type { Config, Operation, Simulation, Snapshot } from "./types";

export function simulate(input: Config): Simulation {
  const config = parseConfig(input);
  const layers = modelFor(config);
  if (config.strategy === "fsdp") validateLayerSharding(layers, config.ranks);
  const compilers = {
    ddp: ddpPlan,
    fsdp: fsdpPlan,
    tp: tpPlan,
    pp: ppPlan,
    ep: epPlan,
  };
  const plan = compilers[config.strategy](config, layers);
  const { operations, events } = schedule(plan.operations, plan.initial);
  const snapshots = replayMemory(config, plan.allocations, operations, events);
  return {
    config,
    layers,
    allocations: plan.allocations,
    operations,
    events,
    snapshots,
    duration: Math.max(0, ...operations.map((operation) => operation.end)),
  };
}

export function snapshotAtEvent(run: Simulation, eventIndex: number): Snapshot {
  const index = Number.isFinite(eventIndex)
    ? Math.min(run.snapshots.length - 1, Math.max(0, Math.floor(eventIndex)))
    : 0;
  return run.snapshots[index];
}

export function snapshotAtTime(run: Simulation, time: number): Snapshot {
  if (!Number.isFinite(time) || time < 0) return run.snapshots[0];
  let low = 0;
  let high = run.snapshots.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (run.snapshots[middle].time <= time) low = middle;
    else high = middle - 1;
  }
  return run.snapshots[low];
}

export function operationStatus(
  run: Simulation,
  operation: Operation | string,
  time: number,
): "ready" | "waiting" | "executing" | "complete" {
  const op =
    typeof operation === "string"
      ? run.operations.find((item) => item.id === operation)
      : operation;
  if (!op) throw new Error("Unknown operation.");
  if (time >= op.end) return "complete";
  if (time >= op.start) return "executing";
  if (time >= op.inputReadyAt) return "ready";
  return "waiting";
}
