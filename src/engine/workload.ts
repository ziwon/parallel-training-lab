import type { Config, Strategy } from "./types";

export const strategies: Strategy[] = ["ddp", "fsdp", "tp", "pp", "ep"];
export const EXPERT_COUNT = 8;
export const REPRESENTATIVE_WIDTH = 8;
export const expertForSample = (sampleId: number, seed: number) =>
  (sampleId * 5 + seed) % EXPERT_COUNT;
export const expertOwner = (expert: number, ranks: number) =>
  Math.floor((expert * ranks) / EXPERT_COUNT);
export const stageOwner = (layer: number, ranks: number) =>
  Math.floor((layer * ranks) / 4);
export const workloadName = (strategy: Strategy) =>
  strategy === "ep"
    ? "Top-1 MoE · 8 linear experts"
    : "Dense · 4 sequential linear layers";
export function parameterIndices(
  strategy: Strategy,
  layer: number,
  rank: number,
  ranks: number,
  width = REPRESENTATIVE_WIDTH,
): number[] {
  return Array.from({ length: width * width }, (_, i) => i).filter((index) => {
    if (strategy === "ddp") return true;
    if (strategy === "pp") return stageOwner(layer, ranks) === rank;
    if (strategy === "ep") return expertOwner(layer, ranks) === rank;
    if (strategy === "fsdp")
      return Math.floor((index * ranks) / (width * width)) === rank;
    const coordinate =
      layer % 2 === 0 ? index % width : Math.floor(index / width);
    return Math.floor((coordinate * ranks) / width) === rank;
  });
}
export function batchDescription(config: Config): string {
  const { strategy, globalBatch: batch, ranks, microbatches } = config;
  if (strategy === "tp")
    return `${batch} samples × 1 data replica · all ${ranks} tensor ranks cooperate`;
  if (strategy === "pp")
    return `${batch} samples = ${batch / microbatches} per microbatch × ${microbatches} microbatches × 1 data replica`;
  if (strategy === "ep")
    return `${batch} tokens = ${batch / ranks} source tokens × ${ranks} ranks · one expert per token`;
  return `${batch} samples = ${batch / ranks} local × ${ranks} data ranks × 1 accumulation step`;
}
