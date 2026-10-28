import { describe, expect, it } from "vitest";
import { defaultConfig } from "../src/engine/config";
import { simulate } from "../src/engine/simulation";
import {
  createFixture,
  executeNumerics,
  getNumericalSnapshot,
  lossFor,
  referenceStep,
  type NumericalFixture,
} from "../src/engine/numerics";
import {
  expertForSample,
  parameterIndices,
  stageOwner,
} from "../src/engine/workload";
import type { Config, RankCount } from "../src/engine/types";

const runNumerics = (config: Partial<Config>) =>
  executeNumerics(simulate({ ...defaultConfig, ...config }));
const assertEquivalent = (result: ReturnType<typeof executeNumerics>) => {
  expect(result.maxGradientError).toBeLessThan(result.tolerance);
  expect(result.maxParameterError).toBeLessThan(result.tolerance);
  expect(result.lossBefore).toBeCloseTo(result.reference.lossBefore, 12);
  expect(result.lossAfter).toBeCloseTo(result.reference.lossAfter, 12);
  for (
    let layer = 0;
    layer < result.fixture.initialParameters.length;
    layer++
  ) {
    const counts = Array<number>(64).fill(0);
    for (const owned of result.parameterIndicesByRank)
      for (const index of owned[layer]) counts[index]++;
    expect(counts).toEqual(Array(64).fill(1));
  }
};

describe("tensor-parallel matrix arithmetic", () => {
  for (const ranks of [1, 2, 4, 8] as RankCount[])
    it(`reconstructs real column and row matrix partitions on ${ranks} ranks`, () => {
      const result = runNumerics({ strategy: "tp", ranks });
      assertEquivalent(result);
      expect(result.fixture).toEqual(
        createFixture(defaultConfig.seed, defaultConfig.globalBatch),
      );
      for (const samples of result.sampleIdsByRank)
        expect(samples).toEqual(
          result.fixture.samples.map((sample) => sample.id),
        );
      result.parameterIndicesByRank.forEach((layers, rank) =>
        layers.forEach((indices, layer) => {
          expect(indices).toEqual(parameterIndices("tp", layer, rank, ranks));
          const coordinates = new Set(
            indices.map((index) =>
              layer % 2 === 0 ? index % 8 : Math.floor(index / 8),
            ),
          );
          expect(coordinates.size).toBe(8 / ranks);
        }),
      );
    });

  it("requires SUM of partial tensors and rejects substituted Average reductions", () => {
    const run = simulate({ ...defaultConfig, strategy: "tp" });
    expect(
      run.operations.filter((operation) => operation.kind === "all-reduce"),
    ).toHaveLength(4);
    const malformed = {
      ...run,
      operations: run.operations.map((operation) =>
        operation.phaseTag === "tp-forward-sum"
          ? { ...operation, reduction: "average" as const }
          : operation,
      ),
    };
    expect(() => executeNumerics(malformed)).toThrow(/SUM/);
  });
});

describe("GPipe microbatch arithmetic", () => {
  for (const ranks of [1, 2, 4] as RankCount[])
    for (const microbatches of [1, 2, 4, 8] as const)
      it(`accumulates the unchanged global-batch gradient on ${ranks} stages and ${microbatches} microbatches`, () => {
        const result = runNumerics({ strategy: "pp", ranks, microbatches });
        assertEquivalent(result);
        result.parameterIndicesByRank.forEach((layers, rank) =>
          layers.forEach((indices, layer) =>
            expect(indices.length).toBe(
              stageOwner(layer, ranks) === rank ? 64 : 0,
            ),
          ),
        );
      });

  it("marks pipeline gradients ready only after every microbatch contributes", () => {
    const run = simulate({ ...defaultConfig, strategy: "pp", microbatches: 4 });
    const result = executeNumerics(run);
    const backward = (microbatch: number) =>
      run.events.find((event) => {
        const operation = run.operations.find(
          (candidate) => candidate.id === event.operation,
        );
        return (
          event.phase === "end" &&
          operation?.kind === "backward" &&
          operation.layers[0] === 3 &&
          operation.microbatch === microbatch
        );
      })!;
    const first = getNumericalSnapshot(result, backward(3).index).rankStates[3];
    expect(first.gradients[3]).not.toBeNull();
    expect(first.synchronizedLayers).not.toContain(3);
    const last = getNumericalSnapshot(result, backward(0).index).rankStates[3];
    expect(last.synchronizedLayers).toContain(3);
  });

  it("computes the same dense problem as DDP and TP across seeds", () => {
    for (const seed of [0, 0xffffffff]) {
      const results = ["ddp", "fsdp", "tp", "pp"].map((strategy) =>
        runNumerics({
          strategy: strategy as Config["strategy"],
          seed,
          globalBatch: 8,
        }),
      );
      for (const result of results) {
        expect(result.fixture).toEqual(results[0].fixture);
        for (let layer = 0; layer < 4; layer++)
          result.reconstructedParameters[layer].forEach((value, index) =>
            expect(value).toBeCloseTo(
              results[0].reconstructedParameters[layer][index],
              12,
            ),
          );
      }
    }
  });
});

describe("fixed top-1 expert arithmetic", () => {
  it("agrees with independently calculated routed scalar gradients", () => {
    const fixture: NumericalFixture = {
      seed: 0,
      width: 1,
      kind: "moe",
      initialParameters: Array.from({ length: 8 }, (_, expert) => [expert + 1]),
      samples: [
        { id: 0, inputs: [2], targets: [1] },
        { id: 1, inputs: [3], targets: [2] },
      ],
      description: "Token 0 visits expert 0 and token 1 visits expert 5.",
    };
    const result = referenceStep(fixture, 0.01);
    expect(result.lossBefore).toBe(64.25);
    expect(result.gradients).toEqual([[1], [0], [0], [0], [0], [24], [0], [0]]);
  });

  for (const ranks of [1, 2, 4, 8] as RankCount[])
    it(`dispatches and returns tokens through their owned experts on ${ranks} ranks`, () => {
      const result = runNumerics({ strategy: "ep", ranks });
      assertEquivalent(result);
      expect(result.sampleIdsByRank.flat()).toEqual(
        result.fixture.samples.map((sample) => sample.id),
      );
      expect(result.fixture.kind).toBe("moe");
      expect(result.fixture.initialParameters).toHaveLength(8);
      result.parameterIndicesByRank.forEach((groups) =>
        groups.forEach((indices) => expect([0, 64]).toContain(indices.length)),
      );
    });

  it("keeps unused expert parameters unchanged and uses global normalization", () => {
    const result = runNumerics({
      strategy: "ep",
      ranks: 1,
      globalBatch: 1,
      seed: 0,
    });
    assertEquivalent(result);
    for (let expert = 1; expert < 8; expert++) {
      expect(result.synchronizedGradients[0][expert]).toEqual(
        Array(64).fill(0),
      );
      expect(result.reconstructedParameters[expert]).toEqual(
        result.fixture.initialParameters[expert],
      );
    }
  });

  it("matches finite differences for routed and empty experts", () => {
    const fixture = createFixture(17, 3, "moe");
    const reference = referenceStep(fixture, 0.01);
    const routes = fixture.samples.map((sample) =>
      expertForSample(sample.id, fixture.seed),
    );
    const epsilon = 1e-5;
    for (let expert = 0; expert < 8; expert++)
      for (const entry of [0, 13, 63]) {
        const above = fixture.initialParameters.map((matrix) => [...matrix]);
        const below = fixture.initialParameters.map((matrix) => [...matrix]);
        above[expert][entry] += epsilon;
        below[expert][entry] -= epsilon;
        expect(reference.gradients[expert][entry]).toBeCloseTo(
          (lossFor(above, fixture.samples, routes) -
            lossFor(below, fixture.samples, routes)) /
            (2 * epsilon),
          10,
        );
      }
  });
});

describe("event-driven updates in every strategy", () => {
  for (const strategy of ["tp", "pp", "ep"] as const)
    it(`keeps ${strategy} parameters unchanged until its optimizer completes and replays exactly`, () => {
      const run = simulate({ ...defaultConfig, strategy });
      const result = executeNumerics(run);
      expect(executeNumerics(run)).toEqual(result);
      const firstAdam = run.events.find(
        (event) =>
          event.phase === "end" &&
          run.operations.find((operation) => operation.id === event.operation)
            ?.kind === "adam",
      )!;
      const before = getNumericalSnapshot(result, firstAdam.index - 1);
      const after = getNumericalSnapshot(result, firstAdam.index);
      expect(
        before.rankStates.every((rank) => rank.updatedLayers.length === 0),
      ).toBe(true);
      expect(
        after.rankStates.some((rank) => rank.updatedLayers.length > 0),
      ).toBe(true);
      expect(getNumericalSnapshot(result, firstAdam.index - 1)).toEqual(before);
      expect(
        result.history
          .at(-1)!
          .rankStates.every((rank) =>
            rank.gradients.every((gradient) => gradient === null),
          ),
      ).toBe(true);
    });
});
