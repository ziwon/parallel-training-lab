import { describe, expect, it } from "vitest";
import {
  adamFirstStep,
  createFixture,
  executeNumerics,
  getNumericalSnapshot,
  lossFor,
  partitionSamples,
  referenceStep,
  type NumericalFixture,
} from "../src/engine/numerics";
import { defaultConfig } from "../src/engine/config";
import { simulate } from "../src/engine/simulation";
import type { Config, RankCount } from "../src/engine/types";

describe("independent numerical fixtures", () => {
  it("agrees with the hand calculation y=3×2, target=1, loss=12.5, gradient=10", () => {
    const fixture: NumericalFixture = {
      seed: 0,
      width: 1,
      initialParameters: [[3]],
      samples: [{ id: 0, inputs: [2], targets: [1] }],
      description: "Hand-checkable scalar linear layer.",
    };
    const result = referenceStep(fixture, 0.1);
    expect(result.lossBefore).toBe(12.5);
    expect(result.gradients).toEqual([[10]]);
    expect(result.firstMoments[0][0]).toBeCloseTo(1, 14);
    expect(result.secondMoments[0][0]).toBeCloseTo(0.1, 14);
    expect(result.parameters[0][0]).toBeCloseTo(2.9000000001, 12);
  });

  it("averages samples and output channels rather than summing losses", () => {
    const fixture: NumericalFixture = {
      seed: 0,
      width: 2,
      initialParameters: [[2, 0, 0, 3]],
      samples: [
        { id: 0, inputs: [1, 2], targets: [0, 0] },
        { id: 1, inputs: [3, 4], targets: [0, 0] },
      ],
      description: "Two channels and two samples.",
    };
    const result = referenceStep(fixture, 0.01);
    expect(result.lossBefore).toBe(27.5);
    expect(result.gradients).toEqual([[5, 10.5, 7, 15]]);
  });

  it("matches a hand-calculated non-diagonal two-layer matrix example", () => {
    const fixture: NumericalFixture = {
      seed: 0,
      width: 2,
      initialParameters: [
        [1, 2, 3, 4],
        [1, 0, 0, 1],
      ],
      samples: [{ id: 0, inputs: [1, 2], targets: [1, 0] }],
      description:
        "The first matrix produces [7, 10], followed by the identity matrix.",
    };
    const result = referenceStep(fixture, 0.01);
    expect(result.lossBefore).toBe(34);
    expect(result.gradients).toEqual([
      [3, 5, 6, 10],
      [21, 35, 30, 50],
    ]);
  });

  it("keeps a zero gradient and parameter unchanged under Adam", () => {
    expect(adamFirstStep(4, 0, 0.01)).toEqual({
      parameter: 4,
      firstMoment: 0,
      secondMoment: 0,
    });
    expect(adamFirstStep(4, -2, 0.01).parameter).toBeCloseTo(4.01, 9);
  });

  it("matches finite differences for every representative parameter", () => {
    const fixture = createFixture(103, 8);
    const reference = referenceStep(fixture, 0.01);
    const epsilon = 1e-5;
    fixture.initialParameters.forEach((row, layer) =>
      row.forEach((_, channel) => {
        const above = fixture.initialParameters.map((values) => [...values]);
        const below = fixture.initialParameters.map((values) => [...values]);
        above[layer][channel] += epsilon;
        below[layer][channel] -= epsilon;
        const finiteDifference =
          (lossFor(above, fixture.samples) - lossFor(below, fixture.samples)) /
          (2 * epsilon);
        expect(reference.gradients[layer][channel]).toBeCloseTo(
          finiteDifference,
          9,
        );
      }),
    );
  });

  it("computes derivatives at zero weights without dividing by the weight", () => {
    const fixture: NumericalFixture = {
      seed: 0,
      width: 1,
      initialParameters: [[0], [3]],
      samples: [{ id: 0, inputs: [2], targets: [1] }],
      description: "A zero weight.",
    };
    expect(referenceStep(fixture, 0.01).gradients).toEqual([[-6], [0]]);
  });

  it("reproduces seeds and partitions every global sample exactly once", () => {
    const fixture = createFixture(42, 32);
    expect(createFixture(42, 32)).toEqual(fixture);
    expect(createFixture(43, 32)).not.toEqual(fixture);
    for (const ranks of [1, 2, 4, 8]) {
      const partitions = partitionSamples(fixture.samples, ranks);
      expect(partitions.every((samples) => samples.length === 32 / ranks)).toBe(
        true,
      );
      expect(partitions.flat().map((sample) => sample.id)).toEqual(
        Array.from({ length: 32 }, (_, id) => id),
      );
    }
    expect(() => partitionSamples(fixture.samples.slice(0, 3), 2)).toThrow();
    expect(() => createFixture(42, 129)).toThrow();
  });
});

describe("arithmetic follows the scheduled training step", () => {
  const modes: Partial<Config>[] = [
    { strategy: "ddp", bucketLayers: 1, overlap: true },
    { strategy: "ddp", bucketLayers: 2, overlap: false },
    { strategy: "fsdp", reshardAfterForward: true, prefetch: 0 },
    { strategy: "fsdp", reshardAfterForward: true, prefetch: 1 },
    { strategy: "fsdp", reshardAfterForward: false, prefetch: 0 },
  ];

  for (const ranks of [1, 2, 4, 8] as RankCount[]) {
    for (const mode of modes) {
      it(`matches the full-batch reference for ${ranks} rank(s), ${JSON.stringify(mode)}`, () => {
        const run = simulate({
          ...defaultConfig,
          ...mode,
          ranks,
          globalBatch: 32,
        });
        const result = executeNumerics(run);
        expect(result.maxGradientError).toBeLessThan(result.tolerance);
        expect(result.maxParameterError).toBeLessThan(result.tolerance);
        expect(result.lossBefore).toBeCloseTo(result.reference.lossBefore, 12);
        expect(result.lossAfter).toBeCloseTo(result.reference.lossAfter, 12);
        expect(result.lossAfter).toBeLessThan(result.lossBefore);
        expect(result.sampleIdsByRank.flat()).toEqual(
          Array.from({ length: 32 }, (_, id) => id),
        );
        if (mode.strategy === "ddp") {
          expect(result.maxReplicaError).toBe(0);
          for (const parameters of result.parametersByRank)
            expect(parameters).toEqual(result.parametersByRank[0]);
        } else {
          for (let layer = 0; layer < 4; layer++) {
            const owners = Array(64).fill(0);
            for (const ranges of result.parameterRangesByRank) {
              const [start, end] = ranges[layer];
              for (let channel = start; channel < end; channel++)
                owners[channel]++;
            }
            expect(owners).toEqual(Array(64).fill(1));
            expect(
              result.parametersByRank.flatMap(
                (parameters) => parameters[layer],
              ),
            ).toEqual(result.reconstructedParameters[layer]);
          }
        }
      });
    }
  }

  it("preserves the training problem when resource costs and virtual storage change", () => {
    const base = executeNumerics(
      simulate({ ...defaultConfig, strategy: "fsdp", prefetch: 1 }),
    );
    const changed = executeNumerics(
      simulate({
        ...defaultConfig,
        strategy: "fsdp",
        prefetch: 1,
        bandwidthGBps: 16,
        computeTFLOPS: 3,
        slowRank: 1,
        slowMultiplier: 3,
        modelScale: 4,
      }),
    );
    expect(changed.fixture).toEqual(base.fixture);
    expect(changed.reconstructedParameters).toEqual(
      base.reconstructedParameters,
    );
    expect(changed.synchronizedGradients).toEqual(base.synchronizedGradients);
  });

  it("keeps strategy equivalence across seeds and global batches", () => {
    for (const seed of [0, 0xffffffff])
      for (const globalBatch of [8, 64]) {
        const ddp = executeNumerics(
          simulate({ ...defaultConfig, seed, globalBatch, strategy: "ddp" }),
        );
        const fsdp = executeNumerics(
          simulate({
            ...defaultConfig,
            seed,
            globalBatch,
            strategy: "fsdp",
            prefetch: 1,
          }),
        );
        expect(fsdp.fixture).toEqual(ddp.fixture);
        expect(fsdp.reconstructedParameters).toEqual(
          ddp.reconstructedParameters,
        );
        expect(fsdp.maxGradientError).toBeLessThan(fsdp.tolerance);
        expect(ddp.maxParameterError).toBeLessThan(ddp.tolerance);
        expect(fsdp.sampleIdsByRank.flat()).toHaveLength(globalBatch);
      }
  });

  it("has no updated numerical values before optimizer completion and replays deterministically", () => {
    const run = simulate({ ...defaultConfig, strategy: "fsdp", prefetch: 1 });
    const result = executeNumerics(run);
    expect(executeNumerics(run)).toEqual(result);
    const firstAdam = run.events.find(
      (event) =>
        event.phase === "end" &&
        run.operations.find((operation) => operation.id === event.operation)
          ?.kind === "adam",
    )!;
    const before = getNumericalSnapshot(result, firstAdam.index - 1);
    expect(
      before.rankStates.every((state) => state.updatedLayers.length === 0),
    ).toBe(true);
    const after = getNumericalSnapshot(result, firstAdam.index);
    expect(
      after.rankStates.some((state) => state.updatedLayers.length > 0),
    ).toBe(true);
    expect(getNumericalSnapshot(result, firstAdam.index - 1)).toEqual(before);
    expect(getNumericalSnapshot(result, firstAdam.index)).toEqual(after);
    expect(
      result.history[0].rankStates.every((state) =>
        state.gradients.every((gradient) => gradient === null),
      ),
    ).toBe(true);
    expect(
      result.history
        .at(-1)!
        .rankStates.every((state) =>
          state.gathered.every((buffer) => buffer === null),
        ),
    ).toBe(true);
  });

  it("refuses a compute event moved before its all-gather", () => {
    const run = simulate({ ...defaultConfig, strategy: "fsdp" });
    const firstForward = run.events.find(
      (event) =>
        event.phase === "start" &&
        run.operations.find((operation) => operation.id === event.operation)
          ?.kind === "forward",
    )!;
    const malformed = {
      ...run,
      events: [
        run.events[0],
        firstForward,
        ...run.events.slice(1).filter((event) => event !== firstForward),
      ],
    };
    expect(() => executeNumerics(malformed)).toThrow(/prerequisite|all-gather/);
  });

  it("refuses to optimize unsynchronized gradients", () => {
    const run = simulate({ ...defaultConfig, strategy: "ddp" });
    const firstAdam = run.events.find(
      (event) =>
        event.phase === "start" &&
        run.operations.find((operation) => operation.id === event.operation)
          ?.kind === "adam",
    )!;
    const malformed = {
      ...run,
      events: [
        run.events[0],
        firstAdam,
        ...run.events.slice(1).filter((event) => event !== firstAdam),
      ],
    };
    expect(() => executeNumerics(malformed)).toThrow(
      /prerequisite|synchronization/,
    );
  });
});
