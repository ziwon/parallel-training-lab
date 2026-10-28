import type { Allocation, Config, Layer } from "./types";
import { expertOwner, stageOwner } from "./workload";

export function shardRange(
  elements: number,
  ranks: number,
  rank: number,
): [number, number] {
  if (
    !Number.isSafeInteger(elements) ||
    elements < 1 ||
    !Number.isInteger(ranks) ||
    ranks < 1 ||
    elements % ranks !== 0 ||
    !Number.isInteger(rank) ||
    rank < 0 ||
    rank >= ranks
  ) {
    throw new Error(
      "Sharding requires positive elements divisible by ranks and a valid owner rank.",
    );
  }
  return [(rank * elements) / ranks, ((rank + 1) * elements) / ranks];
}

export function parameterOwnership(config: Config, layers: Layer[]) {
  if (
    config.strategy === "tp" &&
    layers.some(
      (layer) =>
        (layer.id % 2 === 0 ? layer.outputWidth : layer.inputWidth) %
          config.ranks !==
        0,
    )
  )
    throw new Error(
      "Tensor-parallel matrix dimensions must divide evenly across ranks.",
    );
  return layers.flatMap((layer) =>
    Array.from({ length: config.ranks }, (_, rank) => rank)
      .filter((rank) =>
        config.strategy === "pp"
          ? stageOwner(layer.id, config.ranks) === rank
          : config.strategy === "ep"
            ? expertOwner(layer.id, config.ranks) === rank
            : true,
      )
      .map((rank) => {
        const whole =
          config.strategy === "ddp" ||
          config.strategy === "zero1" ||
          config.strategy === "zero2" ||
          config.strategy === "pp" ||
          config.strategy === "ep";
        const ownership: Allocation["ownership"] = [
          "ddp",
          "zero1",
          "zero2",
        ].includes(config.strategy)
          ? "replica"
          : config.strategy === "pp"
            ? "stage"
            : config.strategy === "ep"
              ? "expert"
              : "shard";
        return {
          layer: layer.id,
          rank,
          ownership,
          range: whole
            ? ([0, layer.elements] as [number, number])
            : config.strategy === "tp"
              ? ([0, layer.elements / config.ranks] as [number, number])
              : shardRange(layer.elements, config.ranks, rank),
          ...(config.strategy === "tp"
            ? {
                matrixSlice: {
                  axis:
                    layer.id % 2 === 0
                      ? ("columns" as const)
                      : ("rows" as const),
                  start:
                    (rank *
                      (layer.id % 2 === 0
                        ? layer.outputWidth
                        : layer.inputWidth)) /
                    config.ranks,
                  end:
                    ((rank + 1) *
                      (layer.id % 2 === 0
                        ? layer.outputWidth
                        : layer.inputWidth)) /
                    config.ranks,
                  inputWidth: layer.inputWidth,
                  outputWidth: layer.outputWidth,
                  copies: config.modelScale,
                },
              }
            : {}),
        };
      }),
  );
}

export function samplePartition(
  globalBatch: number,
  ranks: number,
): number[][] {
  if (
    !Number.isInteger(globalBatch) ||
    !Number.isInteger(ranks) ||
    globalBatch < 1 ||
    ranks < 1 ||
    globalBatch % ranks !== 0
  )
    throw new Error("Global batch must divide evenly across ranks.");
  return Array.from({ length: ranks }, (_, rank) =>
    Array.from(
      { length: globalBatch / ranks },
      (_, i) => (rank * globalBatch) / ranks + i,
    ),
  );
}
