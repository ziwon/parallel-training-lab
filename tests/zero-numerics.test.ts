import { describe, expect, it } from "vitest";
import { defaultConfig } from "../src/engine/config";
import { executeNumerics, getNumericalSnapshot } from "../src/engine/numerics";
import { simulate } from "../src/engine/simulation";
import type {
  PrecisionProfile,
  RankCount,
  ZeroStrategy,
} from "../src/engine/types";

const stages: ZeroStrategy[] = ["ddp", "zero1", "zero2", "zero3"];
const profiles: PrecisionProfile[] = ["fp32", "bf16-master"];
const rankCounts: RankCount[] = [1, 2, 4, 8];

describe("ZeRO numerical equivalence", () => {
  for (const strategy of stages)
    for (const precision of profiles)
      for (const ranks of rankCounts)
        it(`${strategy} ${precision} on ${ranks} ranks matches the full-batch Adam reference`, () => {
          const result = executeNumerics(
            simulate({
              ...defaultConfig,
              strategy,
              precision,
              ranks,
              globalBatch: 32,
            }),
          );
          expect(result.maxGradientError).toBeLessThan(result.tolerance);
          expect(result.maxParameterError).toBeLessThan(result.tolerance);
          expect(result.maxReplicaError).toBe(0);
          expect(result.lossBefore).toBeCloseTo(
            result.reference.lossBefore,
            12,
          );
          expect(result.lossAfter).toBeCloseTo(result.reference.lossAfter, 12);
          expect(result.sampleIdsByRank.flat()).toEqual(
            Array.from({ length: 32 }, (_, index) => index),
          );
          for (let rank = 0; rank < ranks; rank++)
            for (let layer = 0; layer < 4; layer++) {
              const optimizerIndices =
                result.optimizerIndicesByRank[rank][layer];
              expect(optimizerIndices).toHaveLength(
                strategy === "ddp" ? 64 : 64 / ranks,
              );
              expect(result.parameterIndicesByRank[rank][layer]).toHaveLength(
                strategy === "zero3" ? 64 / ranks : 64,
              );
              expect(result.synchronizedGradients[rank][layer]).toHaveLength(
                strategy === "zero2" || strategy === "zero3" ? 64 / ranks : 64,
              );
              expect(result.firstMomentsByRank[rank][layer]).toHaveLength(
                optimizerIndices.length,
              );
              expect(result.secondMomentsByRank[rank][layer]).toHaveLength(
                optimizerIndices.length,
              );
              optimizerIndices.forEach((index, local) => {
                expect(
                  result.firstMomentsByRank[rank][layer][local],
                ).toBeCloseTo(result.reference.firstMoments[layer][index], 12);
                expect(
                  result.secondMomentsByRank[rank][layer][local],
                ).toBeCloseTo(result.reference.secondMoments[layer][index], 12);
              });
              if (strategy !== "zero3")
                expect(result.parametersByRank[rank]).toEqual(
                  result.parametersByRank[0],
                );
            }
          if (strategy !== "ddp")
            for (let layer = 0; layer < 4; layer++)
              expect(
                result.optimizerIndicesByRank.flatMap(
                  (indices) => indices[layer],
                ),
              ).toEqual(Array.from({ length: 64 }, (_, index) => index));
        });

  for (const strategy of stages)
    it(`${strategy} changes storage precision without rounding representative arithmetic`, () => {
      const results = profiles.map((precision) =>
        executeNumerics(simulate({ ...defaultConfig, strategy, precision })),
      );
      expect(results[0].reconstructedParameters).toEqual(
        results[1].reconstructedParameters,
      );
      expect(results[0].synchronizedGradients).toEqual(
        results[1].synchronizedGradients,
      );
      expect(results[0].lossAfter).toBe(results[1].lossAfter);
    });
});

describe("ZeRO replica synchronization", () => {
  for (const strategy of ["zero1", "zero2"] as const) {
    it(`${strategy} updates owner coordinates before gathering them into persistent replicas`, () => {
      const run = simulate({ ...defaultConfig, strategy, ranks: 4 });
      const result = executeNumerics(run);
      const sync = run.operations.find(
        (operation) => operation.purpose === "parameter-sync",
      )!;
      const end = run.events.find(
        (event) => event.operation === sync.id && event.phase === "end",
      )!;
      const before = getNumericalSnapshot(result, end.index - 1);
      const after = getNumericalSnapshot(result, end.index);
      for (const layer of sync.layers)
        for (const state of before.rankStates) {
          state.parameterIndices[layer].forEach((index, local) => {
            const expected = state.optimizerIndices[layer].includes(index)
              ? result.reference.parameters[layer][index]
              : result.fixture.initialParameters[layer][index];
            expect(state.parameters[layer][local]).toBeCloseTo(expected, 12);
          });
          const synced = after.rankStates[state.rank];
          expect(synced.parameters[layer]).toEqual(
            result.reconstructedParameters[layer],
          );
          expect(synced.gathered[layer]).toBeNull();
          expect(synced.parameterSynchronizedLayers).toContain(layer);
        }
    });

    for (const ranks of [1, 4] as const)
      it(`${strategy} rejects missing parameter synchronization on ${ranks} ranks`, () => {
        const run = simulate({ ...defaultConfig, strategy, ranks });
        const syncIds = new Set(
          run.operations
            .filter((operation) => operation.purpose === "parameter-sync")
            .map((operation) => operation.id),
        );
        expect(syncIds.size).toBeGreaterThan(0);
        const withoutSync = {
          ...run,
          operations: run.operations.filter(
            (operation) => !syncIds.has(operation.id),
          ),
          events: run.events.filter(
            (event) => !event.operation || !syncIds.has(event.operation),
          ),
        };
        expect(() => executeNumerics(withoutSync)).toThrow(
          /parameter synchronization/,
        );
      });

    it(`${strategy} rejects synchronization before optimizer owners finish`, () => {
      const run = simulate({ ...defaultConfig, strategy });
      const sync = run.operations.find(
        (operation) => operation.purpose === "parameter-sync",
      )!;
      const start = run.events.find(
        (event) => event.operation === sync.id && event.phase === "start",
      )!;
      const premature = {
        ...run,
        operations: run.operations.map((operation) =>
          operation.id === sync.id
            ? { ...operation, prerequisites: [] }
            : operation,
        ),
        events: [
          run.events[0],
          start,
          ...run.events.slice(1).filter((event) => event !== start),
        ],
      };
      expect(() => executeNumerics(premature)).toThrow(
        /before every owner completed Adam/,
      );
    });
  }
});

describe("ZeRO-3 gathered parameter lifecycle", () => {
  for (const reshardAfterForward of [true, false])
    for (const prefetch of [0, 1] as const)
      it(`releases gathered buffers with reshard=${reshardAfterForward}, prefetch=${prefetch}`, () => {
        const run = simulate({
          ...defaultConfig,
          strategy: "zero3",
          precision: "bf16-master",
          reshardAfterForward,
          prefetch,
        });
        const result = executeNumerics(run);
        expect(result.maxParameterError).toBeLessThan(result.tolerance);
        for (const operation of run.operations.filter(
          (operation) =>
            operation.kind === "backward" || operation.kind === "forward",
        )) {
          const end = run.events.find(
            (event) =>
              event.operation === operation.id && event.phase === "end",
          )!;
          const snapshot = getNumericalSnapshot(result, end.index);
          for (const rank of operation.ranks)
            for (const layer of operation.layers)
              if (operation.kind === "backward" || reshardAfterForward)
                expect(snapshot.rankStates[rank].gathered[layer]).toBeNull();
              else
                expect(snapshot.rankStates[rank].gathered[layer]).toHaveLength(
                  64,
                );
        }
        expect(
          result.history
            .at(-1)!
            .rankStates.every((state) =>
              state.gathered.every((values) => values === null),
            ),
        ).toBe(true);
      });
});
