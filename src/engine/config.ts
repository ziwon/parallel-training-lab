import type { Config } from "./types";

export const defaultConfig: Config = {
  strategy: "ddp",
  ranks: 4,
  globalBatch: 32,
  microbatches: 4,
  seed: 42,
  learningRate: 0.01,
  modelScale: 1,
  bucketLayers: 1,
  overlap: true,
  reshardAfterForward: true,
  prefetch: 0,
  bandwidthGBps: 50,
  latencyUs: 20,
  computeTFLOPS: 10,
  capacityMiB: 512,
  slowRank: null,
  slowMultiplier: 2,
};

export function parseConfig(input: unknown): Config {
  if (!input || typeof input !== "object" || Array.isArray(input))
    throw new Error("Configuration must be an object.");
  const data = input as Record<string, unknown>;
  for (const key of Object.keys(data))
    if (!Object.hasOwn(defaultConfig, key))
      throw new Error(`Unknown configuration field: ${key}.`);
  const value = { ...defaultConfig, ...data };
  if (!["ddp", "fsdp", "tp", "pp", "ep"].includes(value.strategy))
    throw new Error("Strategy must be ddp, fsdp, tp, pp, or ep.");
  if (![1, 2, 4, 8].includes(value.ranks))
    throw new Error("Ranks must be 1, 2, 4, or 8.");
  const integer = (key: keyof Config, low: number, high: number) => {
    const n = value[key];
    if (typeof n !== "number" || !Number.isInteger(n) || n < low || n > high)
      throw new Error(`${key} must be an integer from ${low} to ${high}.`);
  };
  const number = (key: keyof Config, low: number, high: number) => {
    const n = value[key];
    if (typeof n !== "number" || !Number.isFinite(n) || n < low || n > high)
      throw new Error(`${key} must be between ${low} and ${high}.`);
  };
  integer("globalBatch", 1, 128);
  if (
    ["ddp", "fsdp", "ep"].includes(value.strategy) &&
    value.globalBatch % value.ranks !== 0
  )
    throw new Error(
      "Global batch must divide evenly across ranks. No samples are rounded or dropped.",
    );
  if (![1, 2, 4, 8].includes(value.microbatches))
    throw new Error("Pipeline microbatches must be 1, 2, 4, or 8.");
  if (value.strategy === "pp" && value.ranks > 4)
    throw new Error("This four-layer GPipe model supports 1, 2, or 4 stages.");
  if (value.strategy === "pp" && value.globalBatch % value.microbatches !== 0)
    throw new Error(
      "Global batch must divide evenly into pipeline microbatches.",
    );
  integer("seed", 0, 4294967295);
  integer("modelScale", 1, 16);
  if (![1, 2, 4].includes(value.bucketLayers))
    throw new Error("Bucket layers must be 1, 2, or 4.");
  if (![0, 1].includes(value.prefetch))
    throw new Error("Backward prefetch depth must be 0 or 1.");
  for (const key of ["overlap", "reshardAfterForward"] as const)
    if (typeof value[key] !== "boolean")
      throw new Error(`${key} must be a boolean.`);
  number("learningRate", 0.000001, 1);
  number("bandwidthGBps", 0.1, 1000);
  number("latencyUs", 0, 1000);
  number("computeTFLOPS", 0.1, 1000);
  number("capacityMiB", 1, 1048576);
  number("slowMultiplier", 1, 10);
  if (
    value.slowRank !== null &&
    (!Number.isInteger(value.slowRank) ||
      value.slowRank < 0 ||
      value.slowRank >= value.ranks)
  )
    throw new Error("Slow rank must identify a participating rank or be null.");
  return value;
}
