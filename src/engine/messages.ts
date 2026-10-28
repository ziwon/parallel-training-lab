import type { OperationKind } from "./types";

const operationNames: Record<OperationKind, string> = {
  forward: "forward",
  backward: "backward",
  adam: "Adam update",
  "all-gather": "all-gather",
  "all-reduce": "all-reduce",
  "reduce-scatter": "reduce-scatter",
  "send-recv": "send / receive",
  "all-to-all": "all-to-all",
};

export const engineMessages = {
  layers: ["Input projection", "Expand", "Contract", "Output projection"],
  allocations: {
    parameters: "Parameters",
    moments: "Adam moments m + v",
    savedInput: "Saved layer input",
    aliasedGradient: "Full gradient / aliased bucket",
    gradientShard: "Synchronized gradient shard",
    fullGradient: "Full local gradient input",
    gathered: {
      f: "Gathered parameters (forward)",
      b: "Gathered parameters (backward)",
    },
  },
  operation: (kind: OperationKind, layers: number[], rank: number | null) =>
    `${operationNames[kind]} ${layers.map((layer) => `L${layer + 1}`).join(" + ")}${rank === null ? "" : ` · Rank ${rank}`}`,
  modelAssumptions: {
    precision:
      "FP32 storage for parameters, gradients, activations, and both Adam moments. No separate master parameters.",
    arithmetic:
      "Representative JavaScript arithmetic is separate from virtual tensor storage and does not emulate FP32 rounding.",
    optimizer:
      "One bias-corrected Adam update. Moments are initialized eagerly. Gradients are cleared after their optimizer consumer finishes.",
    resources:
      "One compute lane per rank and one serialized communication lane for the process group. Compute and communication may overlap.",
    oneRank:
      "One-rank collectives take zero time and transfer zero bytes. Communication destination buffers remain separately accounted. A one-stage pipeline has no transfers.",
    aliases:
      "DDP buckets alias existing gradients. FSDP gathered parameters and reduced shards are separate buffers. TP reductions, PP transfers, and EP exchanges use explicit separate source and destination storage.",
    exclusions:
      "No master weights, communication scratch, allocator caching, fragmentation, CPU offload, checkpointing, physical topology, or separate root-module policy. Targets and loss scalars are excluded. DDP/FSDP omit transient activation-gradient workspace; TP/PP/EP explicitly count communicated outputs and input gradients.",
  },
} as const;
