import { TOKENS_PER_SAMPLE } from "./model";
import type { Config, OperationKind } from "./types";

export function computeDuration(
  config: Config,
  kind: "forward" | "backward" | "adam",
  elements: number,
  rank: number,
  samples = config.globalBatch / config.ranks,
): number {
  const multiplier = rank === config.slowRank ? config.slowMultiplier : 1;
  if (kind === "adam") return ((elements * 28) / (100 * 1e6)) * multiplier;
  const flops =
    (kind === "forward" ? 2 : 4) * elements * samples * TOKENS_PER_SAMPLE;
  return (flops / (config.computeTFLOPS * 1e9)) * multiplier;
}

export function collectiveCost(
  config: Config,
  kind: Extract<
    OperationKind,
    "all-gather" | "all-reduce" | "reduce-scatter" | "send-recv" | "all-to-all"
  >,
  logicalBytes: number,
) {
  const rounds =
    kind === "send-recv"
      ? 1
      : (config.ranks - 1) * (kind === "all-reduce" ? 2 : 1);
  const transferredBytesPerRank =
    kind === "send-recv"
      ? logicalBytes
      : (logicalBytes * rounds) / config.ranks;
  return {
    duration:
      (rounds * config.latencyUs) / 1000 +
      transferredBytesPerRank / (config.bandwidthGBps * 1e6),
    transferredBytesPerRank,
  };
}
