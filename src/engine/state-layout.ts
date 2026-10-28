import { modelFor } from "./model";
import { shardRange } from "./ownership";
import type { Config, Strategy, ZeroStrategy } from "./types";

export const zeroStrategies = ["ddp", "zero1", "zero2", "zero3"] as const;
export const isZeroStrategy = (strategy: Strategy): strategy is ZeroStrategy =>
  zeroStrategies.some((candidate) => candidate === strategy);
export const isShardedDataParallel = (strategy: Strategy) =>
  strategy === "fsdp" || strategy === "zero3";

export function stateLayout(config: Config) {
  const mixed = config.precision === "bf16-master";
  const stage = zeroStrategies.findIndex(
    (strategy) => strategy === config.strategy,
  );
  return {
    parameterBytes: mixed ? 2 : 4,
    gradientBytes: mixed ? 2 : 4,
    activationBytes: mixed ? 2 : 4,
    masterBytes: mixed ? 4 : 0,
    momentBytes: 8,
    parameterSharded:
      stage === 3 || config.strategy === "fsdp" || config.strategy === "tp",
    gradientSharded:
      stage >= 2 || config.strategy === "fsdp" || config.strategy === "tp",
    optimizerSharded:
      stage >= 1 || config.strategy === "fsdp" || config.strategy === "tp",
  };
}

export function optimizerRange(
  config: Config,
  elements: number,
  rank: number,
): [number, number] {
  return stateLayout(config).optimizerSharded
    ? shardRange(elements, config.ranks, rank)
    : [0, elements];
}

export function zeroStateSummary(config: Config) {
  const elements = modelFor(config).reduce(
    (sum, layer) => sum + layer.elements,
    0,
  );
  return zeroStrategies.map((strategy) => {
    const layout = stateLayout({ ...config, strategy });
    const parameterBytes =
      (elements * layout.parameterBytes) /
      (layout.parameterSharded ? config.ranks : 1);
    const gradientBytes =
      (elements * layout.gradientBytes) /
      (layout.gradientSharded ? config.ranks : 1);
    const masterBytes =
      (elements * layout.masterBytes) /
      (layout.optimizerSharded ? config.ranks : 1);
    const optimizerBytes =
      (elements * layout.momentBytes) /
      (layout.optimizerSharded ? config.ranks : 1);
    const totalBytes =
      parameterBytes + gradientBytes + masterBytes + optimizerBytes;
    return {
      strategy,
      parameterBytes,
      gradientBytes,
      masterBytes,
      optimizerBytes,
      totalBytes,
      bytesPerParameter: totalBytes / elements,
    };
  });
}
