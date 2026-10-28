import type {
  Allocation,
  Config,
  Dependency,
  Layer,
  Operation,
  OperationKind,
} from "../types";
import { collectiveCost, computeDuration } from "../cost";
import { parameterOwnership } from "../ownership";
import { engineMessages } from "../messages";

export interface Plan {
  operations: Operation[];
  allocations: Allocation[];
  initial: string[];
}
export const end = (operation: string): Dependency => ({
  operation,
  phase: "end",
});
export const start = (operation: string): Dependency => ({
  operation,
  phase: "start",
});
export const bufferId = (kind: string, layer: number, rank: number) =>
  `${kind}:${layer}:${rank}`;

export interface OperationOptions {
  elements?: number;
  samples?: number;
  logicalBytes?: number;
  transferredBytesPerRank?: number;
  duration?: number;
  microbatch?: Operation["microbatch"];
  phaseTag?: Operation["phaseTag"];
  reduction?: Operation["reduction"];
  label?: string;
}

export function createPlan(config: Config, layers: Layer[]) {
  const plan: Plan = { operations: [], allocations: [], initial: [] };
  const ranks = Array.from({ length: config.ranks }, (_, rank) => rank);
  const previousCompute = new Map<number, string>();
  let previousCollective: string | null = null;
  let collectiveIndex = 0;

  const allocate = (allocation: Allocation, persistent = false) => {
    plan.allocations.push(allocation);
    if (persistent) plan.initial.push(allocation.id);
    return allocation.id;
  };
  for (const owner of parameterOwnership(config, layers)) {
    const bytes = (owner.range[1] - owner.range[0]) * 4;
    const base = {
      ...owner,
      bytes,
      label: "",
      id: "",
      category: "parameters" as const,
    };
    allocate(
      {
        ...base,
        id: bufferId("p", owner.layer, owner.rank),
        label:
          config.strategy === "tp"
            ? `${owner.layer % 2 === 0 ? "Column" : "Row"} parameter slice`
            : engineMessages.allocations.parameters,
      },
      true,
    );
    allocate(
      {
        ...base,
        id: bufferId("m", owner.layer, owner.rank),
        category: "optimizer",
        bytes: bytes * 2,
        label: engineMessages.allocations.moments,
      },
      true,
    );
  }

  function operation(
    kind: OperationKind,
    id: string,
    layerIds: number[],
    participants: number[],
    prerequisites: Dependency[],
    reads: string[],
    allocations: string[],
    releases: string[],
    options: OperationOptions = {},
  ) {
    const collective = [
      "all-gather",
      "all-reduce",
      "reduce-scatter",
      "send-recv",
      "all-to-all",
    ].includes(kind);
    const elements = layerIds.reduce((sum, l) => sum + layers[l].elements, 0);
    const logicalBytes =
      options.logicalBytes ?? (collective ? elements * 4 : 0);
    const ownedElements =
      kind === "adam" &&
      (config.strategy === "fsdp" || config.strategy === "tp")
        ? elements / config.ranks
        : elements;
    const cost = collective
      ? collectiveCost(
          config,
          kind as
            | "all-gather"
            | "all-reduce"
            | "reduce-scatter"
            | "send-recv"
            | "all-to-all",
          logicalBytes,
        )
      : {
          duration: computeDuration(
            config,
            kind as "forward" | "backward" | "adam",
            options.elements ?? ownedElements,
            participants[0],
            options.samples,
          ),
          transferredBytesPerRank: 0,
        };
    const deps = [...prerequisites];
    if (collective) {
      if (previousCollective) deps.push(end(previousCollective));
      previousCollective = id;
    } else {
      const previous = previousCompute.get(participants[0]);
      if (previous) deps.push(end(previous));
      previousCompute.set(participants[0], id);
    }
    const op: Operation = {
      id,
      kind,
      label:
        options.label ??
        engineMessages.operation(
          kind,
          layerIds,
          collective ? null : participants[0],
        ),
      layers: layerIds,
      ranks: participants,
      prerequisites: deps.filter(
        (dep, index, all) =>
          all.findIndex(
            (other) =>
              other.operation === dep.operation && other.phase === dep.phase,
          ) === index,
      ),
      lane: collective ? "collective" : "compute",
      duration: options.duration ?? cost.duration,
      logicalBytes,
      transferredBytesPerRank:
        options.transferredBytesPerRank ?? cost.transferredBytesPerRank,
      microbatch: options.microbatch,
      phaseTag: options.phaseTag,
      reduction:
        options.reduction ??
        (kind === "all-reduce" || kind === "reduce-scatter"
          ? "average"
          : undefined),
      collectiveIndex: collective ? collectiveIndex++ : null,
      reads,
      allocate: allocations,
      release: releases,
      start: 0,
      end: 0,
      readyByRank: [],
      inputReadyAt: 0,
      resourceReadyAt: 0,
    };
    plan.operations.push(op);
    return op;
  }
  return { plan, ranks, allocate, operation };
}
