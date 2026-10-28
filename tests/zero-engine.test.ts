import { describe, expect, it } from "vitest";
import { defaultConfig, parseConfig } from "../src/engine/config";
import { simulate, snapshotAtEvent } from "../src/engine/simulation";
import {
  stateLayout,
  zeroStateSummary,
  zeroStrategies,
} from "../src/engine/state-layout";
import type { Config } from "../src/engine/types";

const parameterCount = 20_971_520;

describe("ZeRO model state and scheduled storage", () => {
  for (const precision of ["fp32", "bf16-master"] as const)
    for (const ranks of [1, 2, 4, 8] as const)
      it(`${precision} has independent parameter, gradient, master, and moment ownership at ${ranks} ranks`, () => {
        const config: Config = { ...defaultConfig, precision, ranks };
        const summaries = zeroStateSummary(config);
        const p = precision === "fp32" ? 4 : 2;
        const g = p;
        const m = precision === "fp32" ? 0 : 4;
        const expected = [
          p + g + m + 8,
          p + g + (m + 8) / ranks,
          p + (g + m + 8) / ranks,
          (p + g + m + 8) / ranks,
        ];
        expect(summaries.map((row) => row.bytesPerParameter)).toEqual(expected);
        for (const row of summaries) {
          expect(row.totalBytes).toBe(
            parameterCount * expected[zeroStrategies.indexOf(row.strategy)],
          );
          const run = simulate({ ...config, strategy: row.strategy });
          const initial = run.snapshots[0].memory;
          const final = run.snapshots.at(-1)!.memory;
          for (let rank = 0; rank < ranks; rank++) {
            expect(initial[rank].current.parameters).toBe(row.parameterBytes);
            expect(initial[rank].current.optimizer).toBe(row.optimizerBytes);
            expect(initial[rank].current.masterParameters).toBe(
              row.masterBytes,
            );
            expect(initial[rank].current.gradients).toBe(0);
            expect(final[rank].total).toBe(row.totalBytes - row.gradientBytes);
            expect(final[rank].current).toEqual(initial[rank].current);
            expect(final[rank].peakTotal).toBe(
              Math.max(
                ...run.snapshots.map((frame) => frame.memory[rank].total),
              ),
            );
          }
          const layout = stateLayout(run.config);
          for (const allocation of run.allocations) {
            const width =
              allocation.category === "parameters" ||
              allocation.category === "temporary"
                ? p
                : allocation.category === "optimizer"
                  ? 8
                  : allocation.category === "masterParameters"
                    ? 4
                    : allocation.category === "gradients"
                      ? g
                      : layout.activationBytes;
            expect(allocation.bytes).toBe(
              (allocation.range[1] - allocation.range[0]) * width,
            );
          }
          for (const operation of run.operations.filter(
            (op) => op.lane === "collective",
          )) {
            const elements = operation.layers.reduce(
              (sum, layer) => sum + run.layers[layer].elements,
              0,
            );
            expect(operation.logicalBytes).toBe(
              elements * (operation.kind === "all-gather" ? p : g),
            );
            if (ranks === 1) expect(operation.transferredBytesPerRank).toBe(0);
          }
        }
      });

  it("shows exact four-rank theoretical values without treating them as a peak", () => {
    expect(
      zeroStateSummary(defaultConfig).map((row) => row.totalBytes / 1_048_576),
    ).toEqual([320, 200, 140, 80]);
    expect(
      zeroStateSummary({ ...defaultConfig, precision: "bf16-master" }).map(
        (row) => row.totalBytes / 1_048_576,
      ),
    ).toEqual([320, 140, 110, 80]);
    const run = simulate({ ...defaultConfig, strategy: "zero3" });
    expect(run.snapshots.at(-1)!.memory[0].peakTotal).not.toBe(
      zeroStateSummary(defaultConfig)[3].totalBytes,
    );
  });

  it.each(["zero1", "zero2"] as const)(
    "%s updates owner indices before synchronizing persistent replicas",
    (strategy) => {
      const run = simulate({ ...defaultConfig, strategy });
      const syncs = run.operations.filter(
        (op) => op.purpose === "parameter-sync",
      );
      expect(syncs).toHaveLength(4);
      for (const sync of syncs) {
        expect(sync.kind).toBe("all-gather");
        expect(sync.allocate).toEqual([]);
        expect(sync.release).toEqual([]);
        expect(sync.reads).toEqual(
          Array.from({ length: 4 }, (_, rank) => `p:${sync.layers[0]}:${rank}`),
        );
        for (const adam of run.operations.filter((op) => op.kind === "adam"))
          expect(sync.start).toBeGreaterThanOrEqual(adam.end);
      }
      expect(run.operations.some((op) => op.kind === "all-reduce")).toBe(
        strategy === "zero1",
      );
      expect(run.operations.some((op) => op.kind === "reduce-scatter")).toBe(
        strategy === "zero2",
      );
    },
  );

  it("keeps ZeRO-2 local gradients through ReduceScatter and reserves the shard output separately", () => {
    const run = simulate({
      ...defaultConfig,
      strategy: "zero2",
      precision: "bf16-master",
    });
    const reduction = run.operations.find(
      (op) => op.kind === "reduce-scatter",
    )!;
    const before = run.events.find(
      (event) => event.operation === reduction.id && event.phase === "start",
    )!;
    const after = run.events.find(
      (event) => event.operation === reduction.id && event.phase === "end",
    )!;
    expect(snapshotAtEvent(run, before.index).resident).toEqual(
      expect.arrayContaining(["g:3:0", "gs:3:0"]),
    );
    expect(snapshotAtEvent(run, after.index).resident).not.toContain("g:3:0");
    expect(snapshotAtEvent(run, after.index).resident).toContain("gs:3:0");
  });

  it("preserves FSDP's existing FP32 lifecycle as the ZeRO-3 teaching lifecycle", () => {
    for (const reshardAfterForward of [false, true])
      for (const prefetch of [0, 1] as const) {
        const fsdp = simulate({
          ...defaultConfig,
          strategy: "fsdp",
          reshardAfterForward,
          prefetch,
        });
        const zero = simulate({ ...fsdp.config, strategy: "zero3" });
        expect(zero.allocations).toEqual(fsdp.allocations);
        expect(zero.operations).toEqual(fsdp.operations);
        expect(zero.snapshots).toEqual(fsdp.snapshots);
      }
  });

  it("restricts profiles to supported strategies and never adds an FP32 master copy", () => {
    for (const strategy of ["fsdp", "tp", "pp", "ep"] as const)
      expect(() =>
        parseConfig({ ...defaultConfig, strategy, precision: "bf16-master" }),
      ).toThrow(/only/);
    expect(() => parseConfig({ ...defaultConfig, precision: "fp16" })).toThrow(
      /precision/,
    );
    for (const strategy of [
      ...zeroStrategies,
      "fsdp",
      "tp",
      "pp",
      "ep",
    ] as const)
      expect(
        simulate({ ...defaultConfig, strategy }).allocations.some(
          (allocation) => allocation.category === "masterParameters",
        ),
      ).toBe(false);
  });
});
