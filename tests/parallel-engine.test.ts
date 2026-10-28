import { describe, expect, it } from "vitest";
import { defaultConfig, parseConfig } from "../src/engine/config";
import { simulate, snapshotAtEvent } from "../src/engine/simulation";
import { parameterOwnership } from "../src/engine/ownership";
import {
  expertForSample,
  expertOwner,
  parameterIndices,
  stageOwner,
} from "../src/engine/workload";
import type { Config, Simulation } from "../src/engine/types";

const run = (patch: Partial<Config>) =>
  simulate({ ...defaultConfig, ...patch });
const operation = (simulation: Simulation, id: string) =>
  simulation.operations.find((op) => op.id === id)!;
const frame = (simulation: Simulation, id: string, phase: "start" | "end") =>
  snapshotAtEvent(
    simulation,
    simulation.events.find(
      (event) => event.operation === id && event.phase === phase,
    )!.index,
  );

describe("model-parallel plans", () => {
  it("records actual virtual matrix coordinates instead of treating columns as flat shards", () => {
    const simulation = run({ strategy: "tp", ranks: 4 });
    const columns = simulation.allocations.find((a) => a.id === "p:0:1")!;
    expect(columns.matrixSlice).toEqual({
      axis: "columns",
      start: 512,
      end: 1024,
      inputWidth: 1024,
      outputWidth: 2048,
      copies: 1,
    });
    const rows = simulation.allocations.find((a) => a.id === "p:1:1")!;
    expect(rows.matrixSlice).toMatchObject({
      axis: "rows",
      start: 512,
      end: 1024,
    });
    expect(() =>
      parameterOwnership({ ...defaultConfig, strategy: "tp" }, [
        {
          id: 0,
          name: "Invalid columns",
          inputWidth: 2,
          outputWidth: 6,
          elements: 12,
          activationElementsPerSample: 2,
        },
      ]),
    ).toThrow(/dimensions/);
  });
  for (const strategy of ["tp", "pp", "ep"] as const) {
    for (const ranks of [1, 2, 4, 8] as const) {
      if (strategy === "pp" && ranks === 8) continue;
      it(`${strategy} with ${ranks} ranks honors resources, ownership, memory and replay`, () => {
        const simulation = run({
          strategy,
          ranks,
          slowRank: ranks - 1,
          slowMultiplier: 3,
        });
        for (const op of simulation.operations)
          for (const dependency of op.prerequisites)
            expect(op.start).toBeGreaterThanOrEqual(
              operation(simulation, dependency.operation)[dependency.phase],
            );
        const communications = simulation.operations.filter(
          (op) => op.lane === "collective",
        );
        communications.forEach((op, index) => {
          expect(op.collectiveIndex).toBe(index);
          if (index)
            expect(op.start).toBeGreaterThanOrEqual(
              communications[index - 1].end,
            );
          if (ranks === 1) {
            expect(op.duration).toBe(0);
            expect(op.transferredBytesPerRank).toBe(0);
          }
        });
        for (let rank = 0; rank < ranks; rank++) {
          const compute = simulation.operations.filter(
            (op) => op.lane === "compute" && op.ranks[0] === rank,
          );
          for (let index = 1; index < compute.length; index++)
            expect(compute[index].start).toBeGreaterThanOrEqual(
              compute[index - 1].end,
            );
          const final = simulation.snapshots.at(-1)!.memory[rank];
          expect(final.current.activations).toBe(0);
          expect(final.current.temporary).toBe(0);
          expect(final.current.gradients).toBe(0);
          expect(final.current.optimizer).toBe(final.current.parameters * 2);
        }
        const ownership = parameterOwnership(
          simulation.config,
          simulation.layers,
        );
        for (const layer of simulation.layers) {
          expect(
            ownership
              .filter((owner) => owner.layer === layer.id)
              .reduce((sum, owner) => sum + owner.range[1] - owner.range[0], 0),
          ).toBe(layer.elements);
          const indices = Array.from({ length: ranks }, (_, rank) =>
            parameterIndices(strategy, layer.id, rank, ranks),
          )
            .flat()
            .sort((a, b) => a - b);
          expect(indices).toEqual(Array.from({ length: 64 }, (_, i) => i));
        }
        expect(simulate(simulation.config)).toEqual(simulation);
        const before = JSON.stringify(snapshotAtEvent(simulation, 15));
        snapshotAtEvent(simulation, 14);
        expect(JSON.stringify(snapshotAtEvent(simulation, 15))).toBe(before);
      });
    }
  }
  it("rejects empty pipeline stages and fractional microbatches", () => {
    expect(() =>
      parseConfig({ ...defaultConfig, strategy: "pp", ranks: 8 }),
    ).toThrow();
    expect(() =>
      parseConfig({
        ...defaultConfig,
        strategy: "pp",
        globalBatch: 7,
        microbatches: 4,
      }),
    ).toThrow();
  });
  it("does not divide tensor-parallel samples by the tensor degree", () => {
    const one = run({ strategy: "tp", ranks: 1 });
    const four = run({ strategy: "tp", ranks: 4 });
    expect(operation(four, "f:0:0").duration * 4).toBeCloseTo(
      operation(one, "f:0:0").duration,
      12,
    );
    expect(four.allocations.find((a) => a.id === "a:0:0")!.bytes).toBe(
      one.allocations.find((a) => a.id === "a:0:0")!.bytes,
    );
    expect(four.allocations.find((a) => a.id === "a:1:0")!.bytes * 4).toBe(
      one.allocations.find((a) => a.id === "a:1:0")!.bytes,
    );
  });
  it("reduces TP partial activation and input-gradient buffers with SUM", () => {
    const simulation = run({ strategy: "tp" });
    for (const id of ["tp:ar:f:1", "tp:ar:f:3", "tp:ar:b:2", "tp:ar:b:0"]) {
      const op = operation(simulation, id);
      expect(op.reduction).toBe("sum");
      expect(op.reads).toHaveLength(4);
      expect(op.allocate).toHaveLength(4);
      const during = frame(simulation, id, "start");
      for (const input of op.reads) expect(during.resident).toContain(input);
      for (const output of op.allocate)
        expect(during.resident).toContain(output);
      const after = frame(simulation, id, "end");
      for (const input of op.reads) expect(after.resident).not.toContain(input);
    }
    expect(operation(simulation, "f:2:0").start).toBeGreaterThanOrEqual(
      operation(simulation, "tp:ar:f:1").end,
    );
    expect(operation(simulation, "b:1:0").start).toBeGreaterThanOrEqual(
      operation(simulation, "tp:ar:b:2").end,
    );
  });
  it("holds GPipe activations through backward and exposes actual stage transfers", () => {
    const simulation = run({ strategy: "pp", ranks: 4, microbatches: 4 });
    const forwards = simulation.operations.filter(
      (op) => op.kind === "forward",
    );
    const backwards = simulation.operations.filter(
      (op) => op.kind === "backward",
    );
    expect(forwards).toHaveLength(16);
    expect(backwards).toHaveLength(16);
    expect(Math.min(...backwards.map((op) => op.start))).toBeGreaterThanOrEqual(
      Math.max(...forwards.map((op) => op.end)),
    );
    for (const op of backwards) {
      const layer = op.layers[0];
      const saved = `a:${layer}:${stageOwner(layer, 4)}:${op.microbatch}`;
      expect(frame(simulation, op.id, "start").resident).toContain(saved);
      expect(frame(simulation, op.id, "end").resident).not.toContain(saved);
    }
    for (const transfer of simulation.operations.filter(
      (op) => op.kind === "send-recv",
    )) {
      expect(transfer.ranks).toHaveLength(2);
      expect(transfer.logicalBytes).toBeGreaterThan(0);
      expect(transfer.duration).toBeGreaterThan(0);
      const during = frame(simulation, transfer.id, "start");
      expect(during.resident).toContain(transfer.reads[0]);
      expect(during.resident).toContain(transfer.allocate[0]);
    }
    expect(operation(simulation, "f:1:1:0").start).toBeGreaterThanOrEqual(
      operation(simulation, "pp:send:f:0:0").end,
    );
    expect(operation(simulation, "b:0:0:3").start).toBeGreaterThanOrEqual(
      operation(simulation, "pp:send:b:1:3").end,
    );
  });
  it("changes GPipe allocation sizes and event counts without changing the problem", () => {
    const one = run({ strategy: "pp", microbatches: 1 });
    const four = run({ strategy: "pp", microbatches: 4 });
    expect(one.config.globalBatch).toBe(four.config.globalBatch);
    expect(one.layers).toEqual(four.layers);
    expect(operation(one, "f:0:0:0").duration).toBeCloseTo(
      operation(four, "f:0:0:0").duration * 4,
      12,
    );
    expect(one.allocations.find((a) => a.id === "a:0:0:0")!.bytes).toBe(
      four.allocations.find((a) => a.id === "a:0:0:0")!.bytes * 4,
    );
    expect(four.operations.filter((op) => op.kind === "adam")).toHaveLength(4);
  });
  it("routes EP tokens once, retains exchange payloads, and handles empty experts", () => {
    const simulation = run({ strategy: "ep", ranks: 4, globalBatch: 4 });
    expect(simulation.layers).toHaveLength(8);
    const counts = Array.from({ length: 8 }, () => 0);
    for (let sample = 0; sample < 4; sample++)
      counts[expertForSample(sample, simulation.config.seed)]++;
    expect(counts.reduce((a, b) => a + b, 0)).toBe(4);
    expect(counts.filter((count) => count === 0)).toHaveLength(4);
    for (let expert = 0; expert < 8; expert++) {
      const f = operation(simulation, `f:${expert}:${expertOwner(expert, 4)}`);
      expect(f.duration === 0).toBe(counts[expert] === 0);
    }
    const ids = [
      "ep:dispatch",
      "ep:combine",
      "ep:backward-dispatch",
      "ep:backward-combine",
    ];
    expect(
      simulation.operations
        .filter((op) => op.kind === "all-to-all")
        .map((op) => op.id),
    ).toEqual(ids);
    for (const id of ids) {
      const op = operation(simulation, id);
      const during = frame(simulation, id, "start");
      for (const input of op.reads) expect(during.resident).toContain(input);
      for (const output of op.allocate)
        expect(during.resident).toContain(output);
    }
  });
  it.each(["tp", "pp", "ep"] as const)(
    "models bandwidth and capacity for %s",
    (strategy) => {
      const fast = run({ strategy, bandwidthGBps: 100 });
      const slow = run({ strategy, bandwidthGBps: 1 });
      expect(slow.duration).toBeGreaterThan(fast.duration);
      const overflow = run({ strategy, capacityMiB: 1 });
      expect(overflow.snapshots.at(-1)!.memory.every((rank) => rank.oom)).toBe(
        true,
      );
      expect(
        overflow.snapshots.some((snapshot) =>
          snapshot.memory.some((rank) => rank.total > 1048576),
        ),
      ).toBe(true);
    },
  );
});
