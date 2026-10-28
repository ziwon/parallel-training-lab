import { defaultConfig, parseConfig } from "./config";
import { resolveSessionRuns, scenarioIds } from "./scenarios";
import { isZeroStrategy, stateLayout } from "./state-layout";
import type {
  Config,
  ScenarioId,
  SessionConfig,
  Simulation,
  ZeroStrategy,
} from "./types";

export const SCHEMA_VERSION = 3;
export const SIMULATOR_VERSION = "3.0.0";
export const MAX_IMPORT_BYTES = 10 * 1_048_576;

export const modelAssumptions = {
  timeUnit: "modeled milliseconds",
  storageDtypes: {
    parameters: "FP32",
    compute: "FP32",
    gradients: "FP32",
    optimizer: "FP32",
    activations: "FP32",
  },
  arithmetic:
    "JavaScript numbers. FP32 storage labels do not emulate FP32 rounding.",
  accumulationSteps: 1,
  batchPolicy:
    "Fixed global batch. DDP, ZeRO, and FSDP split equal local batches; TP shares samples across matrix partitions; GPipe accumulates microbatches; EP routes source tokens to unique experts.",
  optimizer: {
    name: "Adam",
    steps: 1,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    momentStates: 2,
    masterParameters: false,
  },
  resources:
    "One compute lane per rank and one serialized communication lane for collectives and point-to-point transfers.",
  parameterGroups:
    "One group per dense layer or MoE expert. ZeRO and FSDP use equal flat shards, TP uses row/column partitions, PP owns whole layers, and EP owns whole experts. No padding.",
  ddpBuckets:
    "Gradient bucket views alias gradient tensors and have no second allocation.",
  fsdpBuffers:
    "Gathered parameters and reduced gradient outputs use separate explicit allocations.",
  scratch: "No communication scratch or allocator cache is modeled.",
  activations:
    "Saved inputs follow actual sample, microbatch, or routed-token counts and full/sharded feature layout; each representative token stands for 128 virtual positions. Transfer buffers have explicit lifetimes.",
  scope:
    "One sequential training step. No runtime, hardware topology, root-module special case, precision rounding, checkpointing, offload, or profiler import.",
} as const;

export function assumptionsFor(config: Config) {
  const layout = stateLayout(config);
  const mixed = layout.masterBytes > 0;
  const storage = mixed ? "BF16" : "FP32";
  return {
    ...modelAssumptions,
    precisionProfile: config.precision,
    storageDtypes: {
      parameters: storage,
      compute: "JavaScript numbers",
      gradients: storage,
      optimizer: "FP32",
      masterParameters: mixed ? "FP32" : null,
      activations: storage,
    },
    arithmetic:
      "JavaScript numbers. Storage and traffic widths do not emulate FP32 or BF16 rounding, accumulation, or hardware throughput.",
    optimizer: { ...modelAssumptions.optimizer, masterParameters: mixed },
    stateBytesPerParameter: {
      parameters: layout.parameterBytes,
      gradients: layout.gradientBytes,
      masterParameters: layout.masterBytes,
      adamMoments: layout.momentBytes,
    },
    zeroSchedule:
      "ZeRO-1 averages full gradients; ZeRO-2 reduce-scatters gradients. Both update optimizer-owned indices, then AllGather updated parameters into persistent replicas. ZeRO-3 gathers parameters on demand and releases them according to reshard and prefetch settings.",
    modelStateSummary:
      "Theoretical parameter, gradient, master, and moment footprint with gradients present. Excludes activations and transient buffers; it is not initial, final, or peak allocated memory.",
  };
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error(`${name} must be an object.`);
  return value as Record<string, unknown>;
}

function parseSharedConfig(value: unknown, version: number) {
  const config = record(value, "Configuration");
  const required = Object.keys(defaultConfig).filter(
    (key) =>
      (version >= 3 || key !== "precision") &&
      (version !== 1 || key !== "microbatches"),
  );
  for (const key of required) {
    if (!Object.hasOwn(config, key))
      throw new Error(`Shared configuration is missing ${key}.`);
  }
  if (Object.keys(config).some((key) => !required.includes(key)))
    throw new Error(
      `Version ${version} configuration contains an unsupported field.`,
    );
  if (version === 1 && config.strategy !== "ddp" && config.strategy !== "fsdp")
    throw new Error("Version 1 supports only DDP and FSDP.");
  if (
    version <= 2 &&
    !["ddp", "fsdp", "tp", "pp", "ep"].includes(String(config.strategy))
  )
    throw new Error(`Version ${version} contains an unsupported strategy.`);
  return parseConfig({
    ...config,
    ...(version === 1 ? { microbatches: defaultConfig.microbatches } : {}),
    ...(version <= 2 ? { precision: "fp32" } : {}),
  });
}

function parseSession(value: unknown, version = SCHEMA_VERSION): SessionConfig {
  const session = record(value, "Session");
  const explorer = version === 3 && session.scenario === "zero";
  const fields = explorer
    ? ["config", "compare", "scenario", "zeroPair"]
    : ["config", "compare", "scenario"];
  if (Object.keys(session).some((key) => !fields.includes(key)))
    throw new Error("Session contains an unsupported field.");
  if (typeof session.compare !== "boolean")
    throw new Error("Comparison mode must be a boolean.");
  const config = parseSharedConfig(session.config, version);
  if (explorer) {
    if (session.compare !== true || config.strategy !== "ddp")
      throw new Error(
        "ZeRO Explorer requires comparison mode and a DDP base configuration.",
      );
    if (
      !Array.isArray(session.zeroPair) ||
      session.zeroPair.length !== 2 ||
      !session.zeroPair.every(
        (stage) =>
          typeof stage === "string" &&
          isZeroStrategy(stage as Config["strategy"]),
      )
    )
      throw new Error("ZeRO Explorer requires two supported stages.");
    const parsed: SessionConfig = {
      config,
      compare: true,
      scenario: "zero",
      zeroPair: session.zeroPair as [ZeroStrategy, ZeroStrategy],
    };
    resolveSessionRuns(parsed);
    return parsed;
  }
  if (
    typeof session.scenario !== "string" ||
    !scenarioIds.some((id) => id === session.scenario)
  )
    throw new Error("Unknown comparison scenario.");
  if (
    version === 1 &&
    !["strategies", "reshard", "prefetch", "slow"].includes(session.scenario)
  )
    throw new Error("Version 1 contains an unsupported scenario.");
  if (
    config.precision !== "fp32" ||
    !["ddp", "fsdp", "tp", "pp", "ep"].includes(config.strategy)
  )
    throw new Error(
      "ZeRO stages and mixed storage profiles require ZeRO Explorer.",
    );
  const parsed: SessionConfig = {
    config,
    compare: session.compare,
    scenario: session.scenario as ScenarioId,
  };
  resolveSessionRuns(parsed);
  return parsed;
}

export function encodeSession(session: SessionConfig): string {
  const parameters = new URLSearchParams({
    v: String(SCHEMA_VERSION),
    config: JSON.stringify(parseSession(session)),
  });
  return `?${parameters.toString()}`;
}

export function decodeSession(search: string): SessionConfig {
  if (!search || search === "?")
    return {
      config: { ...defaultConfig },
      compare: false,
      scenario: "strategies",
    };
  if (search.length > 20_000)
    throw new Error("Configuration link is too large.");
  const parameters = new URLSearchParams(search);
  if (
    parameters.getAll("v").length !== 1 ||
    !["1", "2", String(SCHEMA_VERSION)].includes(parameters.get("v") ?? "")
  )
    throw new Error("Unsupported configuration link version.");
  if (
    parameters.getAll("config").length !== 1 ||
    [...parameters.keys()].some((key) => key !== "v" && key !== "config")
  )
    throw new Error(
      "Configuration link contains missing, duplicate, or unsupported fields.",
    );
  let value: unknown;
  try {
    value = JSON.parse(parameters.get("config")!);
  } catch {
    throw new Error("Configuration link contains invalid JSON.");
  }
  return parseSession(value, Number(parameters.get("v")));
}

export function createExport(
  run: Simulation,
  session: SessionConfig,
  numerical: unknown,
) {
  const numericalSummary = Object.fromEntries(
    Object.entries(record(numerical, "Numerical results")).filter(
      ([key]) => key !== "history",
    ),
  );
  return {
    schema: "parallel-viz",
    schemaVersion: SCHEMA_VERSION,
    simulatorVersion: SIMULATOR_VERSION,
    session: parseSession(session),
    inputConfiguration: run.config,
    assumptions: {
      ...assumptionsFor(run.config),
      accumulationSteps:
        run.config.strategy === "pp" ? run.config.microbatches : 1,
      workload:
        run.config.strategy === "ep" ? "moe-8-linear-v2" : "dense-4-linear-v2",
    },
    numericalFixture: {
      name:
        run.config.strategy === "ep"
          ? "Eight fixed-router linear experts"
          : "Four sequential dense linear layers",
      layers: run.config.strategy === "ep" ? 8 : 4,
      width: 8,
      matrixLayout: "Flattened input-major 8 by 8 matrices",
      representativeParameters: run.config.strategy === "ep" ? 512 : 256,
      router:
        run.config.strategy === "ep"
          ? "top-1 expert = (sampleId * 5 + seed) % 8; fixed, no dropping"
          : null,
      seed: run.config.seed,
      loss: "Mean half squared error over samples and eight output coordinates.",
      tolerance: 1e-10,
      virtualStorageIsSeparate: true,
    },
    model: run.layers,
    allocations: run.allocations,
    operations: run.operations,
    events: run.events,
    memoryHistory: run.snapshots.map(({ eventIndex, time, memory }) => ({
      eventIndex,
      time,
      memory,
    })),
    results: {
      duration: run.duration,
      numerical: numericalSummary,
      replay:
        "Numerical boundary snapshots are regenerated from configuration and events; final results and fixture inputs are retained here.",
    },
  };
}

export function importExport(text: string): SessionConfig {
  if (
    text.length > MAX_IMPORT_BYTES ||
    new TextEncoder().encode(text).byteLength > MAX_IMPORT_BYTES
  )
    throw new Error("Import exceeds the 10 MiB limit.");
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("Import is not valid JSON.");
  }
  const data = record(value, "Export");
  if (
    data.schema !== "parallel-viz" ||
    ![1, 2, SCHEMA_VERSION].includes(Number(data.schemaVersion)) ||
    typeof data.schemaVersion !== "number"
  )
    throw new Error("Unsupported export schema.");
  if (data.simulatorVersion !== `${data.schemaVersion}.0.0`)
    throw new Error(
      "This export was created by a different simulator version.",
    );
  for (const field of [
    "model",
    "allocations",
    "operations",
    "events",
    "memoryHistory",
  ]) {
    if (!Array.isArray(data[field]) || data[field].length > 10_000)
      throw new Error(`Export ${field} must be a bounded array.`);
  }
  record(data.assumptions, "Export assumptions");
  record(data.numericalFixture, "Numerical fixture metadata");
  const results = record(data.results, "Export results");
  if (
    typeof results.duration !== "number" ||
    !Number.isFinite(results.duration) ||
    results.duration < 0
  )
    throw new Error("Export duration is invalid.");
  record(results.numerical, "Numerical results");
  parseSharedConfig(data.inputConfiguration, data.schemaVersion);
  return parseSession(data.session, data.schemaVersion);
}
