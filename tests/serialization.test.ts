import { describe, expect, it } from "vitest";
import { defaultConfig } from "../src/engine/config";
import { collectiveLink } from "../src/engine/nccl";
import { executeNumerics } from "../src/engine/numerics";
import {
  resolveSessionRuns,
  scenarioIds,
  scenarioPair,
} from "../src/engine/scenarios";
import {
  createExport,
  decodeSession,
  encodeSession,
  importExport,
  MAX_IMPORT_BYTES,
} from "../src/engine/serialization";
import { simulate } from "../src/engine/simulation";
import type { Operation, SessionConfig, Simulation } from "../src/engine/types";

const session: SessionConfig = {
  config: { ...defaultConfig, seed: 1234 },
  compare: true,
  scenario: "prefetch",
};
const run: Simulation = {
  config: session.config,
  layers: [],
  allocations: [],
  operations: [],
  events: [],
  snapshots: [],
  duration: 2,
};
const operation = (
  kind: Operation["kind"],
  logicalBytes = 32 * 1_048_576,
): Operation => ({
  id: "collective",
  kind,
  label: kind,
  layers: [0],
  ranks: [0, 1, 2, 3],
  prerequisites: [],
  lane: "collective",
  duration: 1,
  logicalBytes,
  transferredBytesPerRank: 24 * 1_048_576,
  collectiveIndex: 0,
  reads: [],
  allocate: [],
  release: [],
  start: 1,
  end: 2,
  readyByRank: [1, 1, 1, 1],
  inputReadyAt: 1,
  resourceReadyAt: 0,
});

describe("versioned sharing", () => {
  it("preserves every input field and the seed", () => {
    expect(decodeSession(encodeSession(session))).toEqual(session);
    expect(encodeSession(decodeSession(encodeSession(session)))).toBe(
      encodeSession(session),
    );
    expect(decodeSession("")).toEqual({
      config: defaultConfig,
      compare: false,
      scenario: "strategies",
    });
  });

  it.each(["tp", "pp", "ep"] as const)(
    "round-trips %s configuration with functional pipeline fields",
    (strategy) => {
      const input: SessionConfig = {
        config: { ...defaultConfig, strategy, microbatches: 8 },
        compare: false,
        scenario: strategy === "ep" ? "experts" : "pipeline",
      };
      const query = encodeSession(input);
      expect(new URLSearchParams(query).get("v")).toBe("3");
      expect(decodeSession(query)).toEqual(input);
    },
  );

  it("migrates complete version 1 DDP/FSDP inputs into version 3 configuration", () => {
    const { microbatches, precision, ...legacyConfig } = session.config;
    expect(precision).toBe("fp32");
    expect(microbatches).toBe(4);
    const legacy = { ...session, config: legacyConfig };
    const query = `?v=1&config=${encodeURIComponent(JSON.stringify(legacy))}`;
    expect(decodeSession(query)).toEqual(session);
    const exported = createExport(run, session, {});
    const oldExport = {
      ...exported,
      schemaVersion: 1,
      simulatorVersion: "1.0.0",
      session: legacy,
      inputConfiguration: legacyConfig,
    };
    expect(importExport(JSON.stringify(oldExport))).toEqual(session);
    expect(() =>
      decodeSession(
        `?v=1&config=${encodeURIComponent(JSON.stringify({ ...legacy, config: { ...legacyConfig, strategy: "tp" } }))}`,
      ),
    ).toThrow(/only DDP/);
    expect(() =>
      decodeSession(
        `?v=2&config=${encodeURIComponent(JSON.stringify(legacy))}`,
      ),
    ).toThrow(/microbatches/);
  });

  it("migrates complete version 2 strategy inputs and rejects new fields in legacy shapes", () => {
    for (const strategy of ["ddp", "fsdp", "tp", "pp", "ep"] as const) {
      const { precision, ...config } = { ...defaultConfig, strategy };
      expect(precision).toBe("fp32");
      const legacy = { config, compare: false, scenario: "strategies" };
      const expected = { ...legacy, config: { ...config, precision: "fp32" } };
      expect(
        decodeSession(
          `?v=2&config=${encodeURIComponent(JSON.stringify(legacy))}`,
        ),
      ).toEqual(expected);
      const exported = createExport(run, session, {});
      expect(
        importExport(
          JSON.stringify({
            ...exported,
            schemaVersion: 2,
            simulatorVersion: "2.0.0",
            session: legacy,
            inputConfiguration: config,
          }),
        ),
      ).toEqual(expected);
      for (const patch of [{ precision: "fp32" }, { strategy: "zero1" }])
        expect(() =>
          decodeSession(
            `?v=2&config=${encodeURIComponent(JSON.stringify({ ...legacy, config: { ...config, ...patch } }))}`,
          ),
        ).toThrow(/unsupported/);
    }
  });

  it("round-trips every ZeRO pair and exports the selected profile assumptions", () => {
    for (const precision of ["fp32", "bf16-master"] as const)
      for (const a of ["ddp", "zero1", "zero2", "zero3"] as const)
        for (const b of ["ddp", "zero1", "zero2", "zero3"] as const) {
          const input: SessionConfig = {
            config: { ...defaultConfig, precision },
            compare: true,
            scenario: "zero",
            zeroPair: [a, b],
          };
          expect(decodeSession(encodeSession(input))).toEqual(input);
          const configs = resolveSessionRuns(input);
          expect(configs.map((config) => config.strategy)).toEqual([a, b]);
          expect(
            configs.every(
              (config) =>
                config.precision === precision &&
                config.seed === input.config.seed &&
                config.globalBatch === input.config.globalBatch &&
                config.ranks === input.config.ranks,
            ),
          ).toBe(true);
          const exported = createExport(
            { ...run, config: configs[1] },
            input,
            {},
          );
          expect(exported.inputConfiguration.strategy).toBe(b);
          expect(exported.assumptions.optimizer.masterParameters).toBe(
            precision === "bf16-master",
          );
          expect(exported.assumptions.storageDtypes.activations).toBe(
            precision === "fp32" ? "FP32" : "BF16",
          );
          expect(importExport(JSON.stringify(exported))).toEqual(input);
        }
  });

  it("rejects malformed Explorer sessions and Explorer fields in standard sessions", () => {
    const valid = {
      config: defaultConfig,
      compare: true,
      scenario: "zero",
      zeroPair: ["ddp", "zero3"],
    };
    for (const patch of [
      { compare: false },
      { zeroPair: ["zero1"] },
      { zeroPair: ["zero1", "fsdp"] },
      { config: { ...defaultConfig, strategy: "zero1" } },
      { scenario: "strategies" },
    ])
      expect(() =>
        decodeSession(
          `?v=3&config=${encodeURIComponent(JSON.stringify({ ...valid, ...patch }))}`,
        ),
      ).toThrow();
    for (const patch of [{ precision: "bf16-master" }, { strategy: "zero2" }])
      expect(() =>
        decodeSession(
          `?v=3&config=${encodeURIComponent(JSON.stringify({ ...session, config: { ...defaultConfig, ...patch } }))}`,
        ),
      ).toThrow(/Explorer/);
  });

  it("exports workload-specific matrix and accumulation metadata", () => {
    const pp = {
      ...session,
      config: {
        ...session.config,
        strategy: "pp" as const,
        microbatches: 8 as const,
      },
      compare: false,
    };
    const dense = createExport({ ...run, config: pp.config }, pp, {});
    expect(dense.numericalFixture.representativeParameters).toBe(256);
    expect(dense.assumptions.accumulationSteps).toBe(8);
    const ep = {
      ...session,
      config: { ...session.config, strategy: "ep" as const },
      compare: false,
    };
    const moe = createExport({ ...run, config: ep.config }, ep, {});
    expect(moe.numericalFixture.representativeParameters).toBe(512);
    expect(moe.numericalFixture.router).toContain("% 8");
    expect(moe.assumptions.workload).not.toBe(dense.assumptions.workload);
  });

  it("keeps supported maximum-rank exports within the import size limit", () => {
    for (const strategy of ["ddp", "fsdp", "tp", "pp", "ep"] as const) {
      const input: SessionConfig = {
        config: {
          ...defaultConfig,
          strategy,
          ranks: strategy === "pp" ? 4 : 8,
          globalBatch: 128,
          microbatches: 8,
        },
        compare: false,
        scenario: "strategies",
      };
      const simulation = simulate(input.config);
      const exported = JSON.stringify(
        createExport(simulation, input, executeNumerics(simulation)),
        null,
        2,
      );
      expect(new TextEncoder().encode(exported).byteLength).toBeLessThan(
        MAX_IMPORT_BYTES,
      );
      expect(importExport(exported)).toEqual(input);
    }
  });

  it("keeps a full eight-rank comparison export importable", () => {
    const input: SessionConfig = {
      config: { ...defaultConfig, ranks: 8, globalBatch: 128 },
      compare: true,
      scenario: "strategies",
    };
    const [a, b] = scenarioPair(input.config, input.scenario).map(simulate);
    const exported = JSON.stringify(
      {
        ...createExport(a, input, executeNumerics(a)),
        comparison: createExport(b, input, executeNumerics(b)),
      },
      null,
      2,
    );
    expect(new TextEncoder().encode(exported).byteLength).toBeLessThan(
      MAX_IMPORT_BYTES,
    );
    expect(importExport(exported)).toEqual(input);
  });

  it("rejects comparison sessions whose alternative has an invalid batch", () => {
    const input: SessionConfig = {
      config: { ...defaultConfig, strategy: "tp", globalBatch: 3 },
      compare: true,
      scenario: "strategies",
    };
    expect(() => encodeSession(input)).toThrow(/batch/i);
    expect(() =>
      decodeSession(`?v=3&config=${encodeURIComponent(JSON.stringify(input))}`),
    ).toThrow(/batch/i);
  });

  it("rejects invalid versions and malformed queries without fallback", () => {
    for (const query of [
      "?v=4&config={}",
      "?v=1&config=nope",
      "?v=1&v=1&config={}",
      "?v=1",
      "?x=1",
      `${encodeSession(session)}&unexpected=1`,
    ]) {
      expect(() => decodeSession(query)).toThrow();
    }
    const missing = { ...session, config: { strategy: "ddp" } };
    expect(() =>
      decodeSession(
        `?v=1&config=${encodeURIComponent(JSON.stringify(missing))}`,
      ),
    ).toThrow(/missing/);
    expect(() =>
      encodeSession({
        ...session,
        config: { ...session.config, globalBatch: 31 },
      }),
    ).toThrow(/batch/i);
    expect(() =>
      encodeSession({
        ...session,
        scenario: "unknown" as "strategies",
      }),
    ).toThrow(/scenario/i);
  });

  it("round-trips export metadata and rebuilds from the validated session only", () => {
    const exported = createExport(run, session, { lossBefore: 1 });
    expect(exported.numericalFixture.seed).toBe(1234);
    expect(exported.assumptions.accumulationSteps).toBe(1);
    expect(importExport(JSON.stringify(exported))).toEqual(session);
    const altered = {
      ...exported,
      events: [
        { time: -1, allocated: ["not-real"], script: "throw new Error()" },
      ],
      results: {
        duration: 3,
        numerical: { malicious: "<script>bad()</script>" },
      },
    };
    expect(importExport(JSON.stringify(altered))).toEqual(session);
  });

  it("rejects incompatible or oversized imports before replay", () => {
    const exported = createExport(run, session, {});
    expect(() =>
      importExport(JSON.stringify({ ...exported, schemaVersion: 4 })),
    ).toThrow(/schema/);
    expect(() =>
      importExport(JSON.stringify({ ...exported, simulatorVersion: "0.1.0" })),
    ).toThrow(/version/);
    expect(() =>
      importExport(JSON.stringify({ ...exported, events: "execute me" })),
    ).toThrow(/array/);
    expect(() =>
      importExport(
        JSON.stringify({
          ...exported,
          results: { duration: -1, numerical: {} },
        }),
      ),
    ).toThrow(/duration/);
    expect(() =>
      importExport(
        JSON.stringify({
          ...exported,
          results: { duration: 1, numerical: null },
        }),
      ),
    ).toThrow(/Numerical/);
    expect(() => importExport(" ".repeat(MAX_IMPORT_BYTES + 1))).toThrow(
      /10 MiB/,
    );
    expect(() => importExport("{")).toThrow(/JSON/);
  });

  it.each(["ddp", "fsdp", "tp", "pp", "ep"] as const)(
    "recreates the complete %s run from an exported session",
    (strategy) => {
      const input: SessionConfig = {
        ...session,
        config: { ...session.config, strategy },
      };
      const original = simulate(input.config);
      const numerical = executeNumerics(original);
      const exported = createExport(original, input, numerical);
      const restored = simulate(importExport(JSON.stringify(exported)).config);
      expect(restored).toEqual(original);
      expect(executeNumerics(restored).reconstructedParameters).toEqual(
        numerical.reconstructedParameters,
      );
      expect(exported.memoryHistory).toHaveLength(original.events.length);
    },
  );
});

describe("fixed-problem comparisons", () => {
  it.each(scenarioIds)("preserves the training problem for %s", (scenario) => {
    const [a, b] = scenarioPair(session.config, scenario);
    for (const key of [
      "globalBatch",
      "seed",
      "learningRate",
      "modelScale",
    ] as const) {
      expect(a[key]).toBe(session.config[key]);
      expect(b[key]).toBe(session.config[key]);
    }
    expect(a.strategy === "ep").toBe(b.strategy === "ep");
    if (scenario === "experts")
      expect([a.ranks, b.ranks]).toEqual([1, session.config.ranks]);
    else expect(a.ranks).toBe(b.ranks);
  });

  it("isolates the intended reshard, prefetch, and slow-rank changes", () => {
    const reshard = scenarioPair(defaultConfig, "reshard");
    expect(reshard.map((c) => c.reshardAfterForward)).toEqual([true, false]);
    expect(reshard.map((c) => c.prefetch)).toEqual([0, 0]);
    const prefetch = scenarioPair(defaultConfig, "prefetch");
    expect(prefetch.map((c) => c.prefetch)).toEqual([0, 1]);
    expect(
      prefetch.every((c) => c.strategy === "fsdp" && c.reshardAfterForward),
    ).toBe(true);
    const slow = scenarioPair(defaultConfig, "slow");
    expect(slow.map((c) => c.slowRank)).toEqual([null, 3]);
    expect(slow[1].slowMultiplier).toBe(2);
    expect(
      scenarioPair({ ...defaultConfig, ranks: 1 }, "slow")[1].slowRank,
    ).toBe(0);
  });
});

describe("NCCL Lab integration", () => {
  it.each([
    ["all-reduce", "AllReduce", "per-rank buffer"],
    ["all-gather", "AllGather", "total gathered output"],
    ["reduce-scatter", "ReduceScatter", "total input before scattering"],
  ] as const)(
    "sends the correct total logical size for %s",
    (kind, name, meaning) => {
      const link = collectiveLink(operation(kind), defaultConfig);
      const url = new URL(link.url!);
      expect(url.origin).toBe("https://nccl-lab.pages.dev");
      expect(url.searchParams.get("op")).toBe(name);
      expect(url.searchParams.get("size")).toBe("32");
      expect(url.searchParams.get("n")).toBe("4");
      expect(url.searchParams.get("alg")).toBe("Ring");
      expect(url.searchParams.get("red")).toBe(
        kind === "all-gather" ? "Sum" : "Average",
      );
      expect(url.searchParams.get("bw")).toBe("50");
      expect([...url.searchParams.keys()].sort()).toEqual([
        "alg",
        "bw",
        "n",
        "op",
        "red",
        "root",
        "size",
      ]);
      expect(link.note).toContain(meaning);
      expect(link.note).toContain("illustrative scalar samples");
    },
  );

  it("uses SUM for TP partial results and a documented illustrative AllToAll for EP", () => {
    const tp = collectiveLink(
      {
        ...operation("all-reduce"),
        reduction: "sum",
        phaseTag: "tp-forward-sum",
      },
      { ...defaultConfig, strategy: "tp" },
    );
    expect(new URL(tp.url!).searchParams.get("red")).toBe("Sum");
    const ep = collectiveLink(operation("all-to-all"), {
      ...defaultConfig,
      strategy: "ep",
    });
    const url = new URL(ep.url!);
    expect(url.searchParams.get("op")).toBe("AllToAll");
    expect(url.searchParams.get("alg")).toBe("Naive");
    expect(url.searchParams.get("size")).toBe("32");
    expect(ep.note).toContain("maximum modeled rank input");
    expect(ep.note).toContain("unequal peer counts");
    const transfer = collectiveLink(operation("send-recv"), {
      ...defaultConfig,
      strategy: "pp",
    });
    expect(transfer.url).toBeNull();
    expect(transfer.note).toContain("point-to-point");
  });

  it("converts bytes exactly without treating them as MiB or a shard size", () => {
    expect(
      new URL(
        collectiveLink(operation("all-gather", 4096), defaultConfig).url!,
      ).searchParams.get("size"),
    ).toBe(String(4096 / 1_048_576));
  });

  it("explains unsupported bandwidth presets and the one-rank no-op", () => {
    const custom = collectiveLink(operation("all-reduce"), {
      ...defaultConfig,
      bandwidthGBps: 37,
    });
    expect(new URL(custom.url!).searchParams.has("bw")).toBe(false);
    expect(custom.note).toContain("default 50 GB/s");
    expect(
      collectiveLink(operation("all-gather"), { ...defaultConfig, ranks: 1 })
        .url,
    ).toBeNull();
    expect(
      collectiveLink(operation("all-gather"), { ...defaultConfig, ranks: 1 })
        .note,
    ).toContain("no-op");
    expect(collectiveLink(operation("forward"), defaultConfig).url).toBeNull();
    expect(
      collectiveLink(operation("all-gather", 0), defaultConfig).url,
    ).toBeNull();
  });
});
