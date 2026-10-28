import { describe, expect, it } from "vitest";
import { defaultConfig, parseConfig } from "../src/engine/config";
import { modelFor, validateLayerSharding } from "../src/engine/model";
import { replayMemory } from "../src/engine/memory";
import { samplePartition, shardRange } from "../src/engine/ownership";
import { schedule } from "../src/engine/scheduler";
import {
  operationStatus,
  simulate,
  snapshotAtEvent,
  snapshotAtTime,
} from "../src/engine/simulation";
import type { Config, Simulation } from "../src/engine/types";

const fsdp = (patch: Partial<Config> = {}) =>
  simulate({ ...defaultConfig, strategy: "fsdp", ...patch });
const op = (run: Simulation, id: string) =>
  run.operations.find((operation) => operation.id === id)!;

describe("configuration and ownership", () => {
  it("preserves the global batch and partitions every sample exactly once", () => {
    for (const ranks of [1, 2, 4, 8] as const) {
      const config = parseConfig({ ...defaultConfig, ranks });
      const partition = samplePartition(config.globalBatch, ranks);
      expect(partition.flat()).toEqual(Array.from({ length: 32 }, (_, i) => i));
      expect(partition.every((samples) => samples.length === 32 / ranks)).toBe(
        true,
      );
      expect(config.globalBatch).toBe(32);
    }
    expect(() => parseConfig({ ...defaultConfig, globalBatch: 15 })).toThrow(
      /divide evenly/,
    );
  });
  it("rejects invalid imported types and limits rather than silently rounding", () => {
    for (const patch of [
      { ranks: 3 },
      { ranks: "4" },
      { modelScale: 1.5 },
      { prefetch: 2 },
      { globalBatch: 0 },
      { capacityMiB: NaN },
      { seed: -1 },
      { slowRank: 4 },
      { overlap: "false" },
      { constructor: 1 },
    ]) {
      expect(() => parseConfig({ ...defaultConfig, ...patch })).toThrow();
    }
    const layers = modelFor(defaultConfig);
    layers[0].elements = 15;
    expect(() => validateLayerSharding(layers, 4)).toThrow(/whole number/);
  });
  it("partitions each FSDP parameter into exactly one contiguous owner range", () => {
    const run = fsdp();
    for (const layer of run.layers) {
      const ranges = Array.from({ length: run.config.ranks }, (_, rank) =>
        shardRange(layer.elements, run.config.ranks, rank),
      );
      expect(ranges[0][0]).toBe(0);
      expect(ranges.at(-1)![1]).toBe(layer.elements);
      for (let i = 1; i < ranges.length; i++)
        expect(ranges[i - 1][1]).toBe(ranges[i][0]);
    }
  });
});

describe("resource scheduling", () => {
  it.each(["ddp", "fsdp"] as const)(
    "honors every %s prerequisite and serializes each lane",
    (strategy) => {
      const run = simulate({
        ...defaultConfig,
        strategy,
        prefetch: 1,
        slowRank: 2,
      });
      for (const operation of run.operations)
        for (const dependency of operation.prerequisites)
          expect(operation.start).toBeGreaterThanOrEqual(
            op(run, dependency.operation)[dependency.phase],
          );
      const collectives = run.operations.filter(
        (operation) => operation.lane === "collective",
      );
      collectives.forEach((operation, index) => {
        expect(operation.collectiveIndex).toBe(index);
        if (index > 0)
          expect(operation.start).toBeGreaterThanOrEqual(
            collectives[index - 1].end,
          );
      });
      for (let rank = 0; rank < run.config.ranks; rank++) {
        const compute = run.operations.filter(
          (operation) =>
            operation.lane === "compute" && operation.ranks[0] === rank,
        );
        for (let i = 1; i < compute.length; i++)
          expect(compute[i].start).toBeGreaterThanOrEqual(compute[i - 1].end);
      }
    },
  );
  it("overlaps ready DDP buckets with remaining backward work", () => {
    const run = simulate(defaultConfig);
    const first = op(run, "ar:0");
    expect(first.start).toBeGreaterThanOrEqual(op(run, "b:3:0").end);
    expect(first.start).toBeLessThan(op(run, "b:0:0").end);
    expect(op(run, "b:2:0").start).toBeLessThan(first.end);
    const serial = simulate({ ...defaultConfig, overlap: false });
    expect(op(serial, "ar:0").start).toBeGreaterThanOrEqual(
      op(serial, "b:0:0").end,
    );
    expect(serial.duration).toBeGreaterThan(run.duration);
  });
  it("never consumes gathered parameters before the all-gather completes", () => {
    for (const prefetch of [0, 1] as const) {
      const run = fsdp({ prefetch });
      for (const operation of run.operations.filter(
        (operation) =>
          operation.kind === "forward" || operation.kind === "backward",
      )) {
        const gather = op(
          run,
          `ag:${operation.kind === "forward" ? "f" : "b"}:${operation.layers[0]}`,
        );
        expect(operation.start).toBeGreaterThanOrEqual(gather.end);
      }
    }
  });
  it("models a slow participant without slowing other local compute durations", () => {
    const fast = simulate(defaultConfig);
    const slow = simulate({ ...defaultConfig, slowRank: 3, slowMultiplier: 3 });
    expect(op(slow, "b:3:0").duration).toBe(op(fast, "b:3:0").duration);
    expect(op(slow, "ar:0").start).toBeGreaterThan(op(slow, "b:3:0").end);
    expect(op(slow, "ar:0").readyByRank[3]).toBeGreaterThan(
      op(slow, "ar:0").readyByRank[0],
    );
    expect(slow.duration).toBeGreaterThan(fast.duration);
  });
  it("changes communication duration when bandwidth changes", () => {
    const fast = fsdp({ bandwidthGBps: 100 });
    const slow = fsdp({ bandwidthGBps: 10 });
    expect(op(slow, "ag:f:0").duration).toBeGreaterThan(
      op(fast, "ag:f:0").duration,
    );
    expect(slow.config.globalBatch).toBe(fast.config.globalBatch);
  });
  it("reports a following layer as waiting for its input before previous compute finishes", () => {
    const run = simulate(defaultConfig);
    expect(operationStatus(run, "f:1:0", 0)).toBe("waiting");
    expect(operationStatus(run, "f:0:0", 0)).toBe("executing");
    expect(operationStatus(run, "f:0:0", run.duration)).toBe("complete");
  });
});

describe("allocation lifetimes and replay", () => {
  it("retains gathered parameters through backward and omits backward all-gathers", () => {
    const retained = fsdp({ reshardAfterForward: false, prefetch: 1 });
    expect(
      retained.operations.filter(
        (operation) => operation.kind === "all-gather",
      ),
    ).toHaveLength(4);
    expect(
      fsdp().operations.filter((operation) => operation.kind === "all-gather"),
    ).toHaveLength(8);
    const forwardEnd = retained.events.find(
      (event) => event.operation === "f:0:0" && event.phase === "end",
    )!;
    expect(snapshotAtEvent(retained, forwardEnd.index).resident).toContain(
      "pgf:0:0",
    );
    const backwardEnd = retained.events.find(
      (event) => event.operation === "b:0:0" && event.phase === "end",
    )!;
    expect(snapshotAtEvent(retained, backwardEnd.index).resident).not.toContain(
      "pgf:0:0",
    );
  });
  it("keeps full gradients until reduce-scatter finishes and counts its output separately", () => {
    const run = fsdp();
    const rs = op(run, "rs:3");
    const start = run.events.find(
      (event) => event.operation === rs.id && event.phase === "start",
    )!;
    const end = run.events.find(
      (event) => event.operation === rs.id && event.phase === "end",
    )!;
    const during = snapshotAtEvent(run, start.index);
    expect(during.resident).toContain("g:3:0");
    expect(during.resident).toContain("gs:3:0");
    expect(snapshotAtEvent(run, end.index).resident).not.toContain("g:3:0");
    expect(snapshotAtEvent(run, end.index).resident).toContain("gs:3:0");
  });
  it("preserves saved inputs until their matching backward completion", () => {
    for (const strategy of ["ddp", "fsdp"] as const) {
      const run = simulate({ ...defaultConfig, strategy });
      for (const backward of run.operations.filter(
        (operation) => operation.kind === "backward",
      )) {
        const id = `a:${backward.layers[0]}:${backward.ranks[0]}`;
        const start = run.events.find(
          (event) => event.operation === backward.id && event.phase === "start",
        )!;
        const end = run.events.find(
          (event) => event.operation === backward.id && event.phase === "end",
        )!;
        expect(snapshotAtEvent(run, start.index).resident).toContain(id);
        expect(snapshotAtEvent(run, end.index).resident).not.toContain(id);
      }
    }
  });
  it("accounts actual prefetched residency while bounding backward gathered groups to two", () => {
    const off = fsdp({ prefetch: 0 });
    const on = fsdp({ prefetch: 1 });
    expect(op(on, "ag:b:2").start).toBeLessThan(op(on, "b:3:0").end);
    expect(op(off, "ag:b:2").start).toBeGreaterThanOrEqual(
      op(off, "b:3:0").end,
    );
    const start = on.events.find(
      (event) => event.operation === "ag:b:2" && event.phase === "start",
    )!;
    const snapshot = snapshotAtEvent(on, start.index);
    expect(snapshot.resident).toContain("pgb:2:0");
    expect(snapshot.resident).toContain("pgb:3:0");
    const allocatedBytes = snapshot.resident
      .map((id) => on.allocations.find((allocation) => allocation.id === id)!)
      .filter((allocation) => allocation.rank === 0)
      .reduce((sum, allocation) => sum + allocation.bytes, 0);
    expect(snapshot.memory[0].total).toBe(allocatedBytes);
    expect(on.snapshots.at(-1)!.memory[0].peak.temporary).toBeGreaterThan(
      off.snapshots.at(-1)!.memory[0].peak.temporary,
    );
    for (const run of [
      on,
      fsdp({ prefetch: 1, slowRank: 3, slowMultiplier: 5 }),
    ])
      for (const frame of run.snapshots)
        for (let rank = 0; rank < run.config.ranks; rank++)
          expect(
            frame.resident.filter(
              (id) => id.startsWith("pgb:") && id.endsWith(`:${rank}`),
            ).length,
          ).toBeLessThanOrEqual(2);
  });
  it("counts DDP bucket views once and clears gradients after optimizer use", () => {
    const run = simulate(defaultConfig);
    expect(
      run.allocations.some(
        (allocation) =>
          allocation.label.includes("bucket") &&
          allocation.category !== "gradients",
      ),
    ).toBe(false);
    const totalParameters = run.layers.reduce(
      (sum, layer) => sum + layer.elements * 4,
      0,
    );
    const last = run.snapshots.at(-1)!;
    expect(last.memory[0].total).toBe(totalParameters * 3);
    expect(last.memory[0].current.gradients).toBe(0);
    expect(last.memory[0].peak.gradients).toBe(totalParameters);
  });
  it("reports capacity violations without clamping or dropping buffers", () => {
    const run = fsdp({ capacityMiB: 1 });
    expect(run.snapshots[0].memory[0].oom).toBe(true);
    expect(run.snapshots[0].memory[0].total).toBeGreaterThan(1048576);
    expect(run.snapshots.at(-1)!.memory[0].oom).toBe(true);
  });
  it.each(["ddp", "fsdp"] as const)(
    "handles one-rank %s with causal zero-duration collectives",
    (strategy) => {
      const run = simulate({
        ...defaultConfig,
        strategy,
        ranks: 1,
        prefetch: 1,
      });
      for (const collective of run.operations.filter(
        (operation) => operation.lane === "collective",
      )) {
        expect(collective.duration).toBe(0);
        expect(collective.transferredBytesPerRank).toBe(0);
        const start = run.events.find(
          (event) =>
            event.operation === collective.id && event.phase === "start",
        )!;
        const end = run.events.find(
          (event) => event.operation === collective.id && event.phase === "end",
        )!;
        expect(start.index).toBeLessThan(end.index);
      }
      expect(run.snapshots.at(-1)!.completed).toHaveLength(
        run.operations.length,
      );
    },
  );
  it("reproduces snapshots when stepping backward and forward, and on fresh simulation", () => {
    const run = fsdp({ prefetch: 1, slowRank: 1 });
    const saved = JSON.stringify(snapshotAtEvent(run, 30));
    snapshotAtEvent(run, 29);
    expect(JSON.stringify(snapshotAtEvent(run, 30))).toBe(saved);
    expect(simulate(run.config)).toEqual(run);
    expect(snapshotAtTime(run, run.duration)).toEqual(run.snapshots.at(-1));
  });
  it("rejects double allocations and premature releases instead of hiding ledger corruption", () => {
    const run = fsdp();
    const duplicate = structuredClone(run.events);
    duplicate[0].allocated.push(duplicate[0].allocated[0]);
    expect(() =>
      replayMemory(run.config, run.allocations, run.operations, duplicate),
    ).toThrow(/allocated twice/);
    const premature = structuredClone(run.events);
    const start = premature.find(
      (event) => event.operation === "f:0:0" && event.phase === "start",
    )!;
    start.released.push("pgf:0:0");
    expect(() =>
      replayMemory(run.config, run.allocations, run.operations, premature),
    ).toThrow(/consumer is executing/);
  });
  it("rejects cyclic operation prerequisites", () => {
    const run = simulate(defaultConfig);
    const operations = structuredClone(run.operations);
    operations[0].prerequisites.push({
      operation: operations.at(-1)!.id,
      phase: "end",
    });
    expect(() => schedule(operations, [])).toThrow(/cycle/);
  });
  it("preserves ledger and prefetch bounds across supported ranks and resource extremes", () => {
    for (const ranks of [1, 2, 4, 8] as const)
      for (const prefetch of [0, 1] as const)
        for (const bandwidthGBps of [0.1, 1000])
          for (const computeTFLOPS of [0.1, 1000]) {
            const run = fsdp({
              ranks,
              prefetch,
              bandwidthGBps,
              computeTFLOPS,
              slowRank: ranks - 1,
              slowMultiplier: 10,
              modelScale: 16,
            });
            expect(run.snapshots.at(-1)!.completed).toHaveLength(
              run.operations.length,
            );
            for (const frame of run.snapshots)
              for (let rank = 0; rank < ranks; rank++) {
                expect(
                  frame.resident.filter(
                    (id) => id.startsWith("pgb:") && id.endsWith(`:${rank}`),
                  ).length,
                ).toBeLessThanOrEqual(prefetch + 1);
                expect(frame.memory[rank].total).toBeGreaterThanOrEqual(0);
              }
          }
  });
});
