export type ZeroStrategy = "ddp" | "zero1" | "zero2" | "zero3";
export type Strategy = ZeroStrategy | "fsdp" | "tp" | "pp" | "ep";
export type PrecisionProfile = "fp32" | "bf16-master";
export type RankCount = 1 | 2 | 4 | 8;
export interface Config {
  strategy: Strategy;
  precision: PrecisionProfile;
  ranks: RankCount;
  globalBatch: number;
  microbatches: 1 | 2 | 4 | 8;
  seed: number;
  learningRate: number;
  modelScale: number;
  bucketLayers: 1 | 2 | 4;
  overlap: boolean;
  reshardAfterForward: boolean;
  prefetch: 0 | 1;
  bandwidthGBps: number;
  latencyUs: number;
  computeTFLOPS: number;
  capacityMiB: number;
  slowRank: number | null;
  slowMultiplier: number;
}
export interface Layer {
  id: number;
  name: string;
  inputWidth: number;
  outputWidth: number;
  elements: number;
  activationElementsPerSample: number;
}
export const categories = [
  "parameters",
  "gradients",
  "optimizer",
  "masterParameters",
  "activations",
  "temporary",
  "scratch",
] as const;
export type Category = (typeof categories)[number];
export type MemoryTotals = Record<Category, number>;
export type OperationKind =
  | "forward"
  | "backward"
  | "adam"
  | "all-gather"
  | "all-reduce"
  | "reduce-scatter"
  | "send-recv"
  | "all-to-all";
export interface Dependency {
  operation: string;
  phase: "start" | "end";
}
export interface Allocation {
  id: string;
  rank: number;
  layer: number;
  category: Category;
  ownership: "replica" | "shard" | "gathered" | "local" | "stage" | "expert";
  label: string;
  bytes: number;
  range: [number, number];
  matrixSlice?: {
    axis: "columns" | "rows";
    start: number;
    end: number;
    inputWidth: number;
    outputWidth: number;
    copies: number;
  };
}
export interface Operation {
  id: string;
  kind: OperationKind;
  label: string;
  layers: number[];
  ranks: number[];
  prerequisites: Dependency[];
  lane: "compute" | "collective";
  duration: number;
  logicalBytes: number;
  transferredBytesPerRank: number;
  collectiveIndex: number | null;
  reads: string[];
  allocate: string[];
  release: string[];
  start: number;
  end: number;
  readyByRank: number[];
  inputReadyAt: number;
  resourceReadyAt: number;
  microbatch?: number;
  phaseTag?:
    | "tp-forward-sum"
    | "tp-backward-sum"
    | "pp-forward-transfer"
    | "pp-backward-transfer"
    | "ep-dispatch"
    | "ep-combine"
    | "ep-backward-dispatch"
    | "ep-backward-combine";
  reduction?: "sum" | "average";
  purpose?: "parameter-sync";
}
export interface SimulationEvent {
  index: number;
  time: number;
  operation: string | null;
  phase: "initialize" | "start" | "end";
  allocated: string[];
  released: string[];
}
export interface RankMemory {
  rank: number;
  current: MemoryTotals;
  peak: MemoryTotals;
  total: number;
  peakTotal: number;
  oom: boolean;
}
export interface Snapshot {
  eventIndex: number;
  time: number;
  resident: string[];
  memory: RankMemory[];
  completed: string[];
  executing: string[];
}
export interface Simulation {
  config: Config;
  layers: Layer[];
  allocations: Allocation[];
  operations: Operation[];
  events: SimulationEvent[];
  snapshots: Snapshot[];
  duration: number;
}
export type ScenarioId =
  | "strategies"
  | "reshard"
  | "prefetch"
  | "slow"
  | "tensor"
  | "pipeline"
  | "experts";
export type SessionConfig =
  | { config: Config; compare: boolean; scenario: ScenarioId; zeroPair?: never }
  | {
      config: Config;
      compare: true;
      scenario: "zero";
      zeroPair: [ZeroStrategy, ZeroStrategy];
    };
