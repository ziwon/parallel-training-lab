import type { Operation, Simulation, SimulationEvent } from "./types";
import {
  isShardedDataParallel,
  isZeroStrategy,
  optimizerRange,
} from "./state-layout";
import {
  EXPERT_COUNT,
  expertForSample,
  expertOwner,
  parameterIndices,
  stageOwner,
} from "./workload";

export const numericalTolerance = 1e-10;
export const fixtureWidth = 8;
export const fixtureLayers = 4;
export const adamSettings = {
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  step: 1,
} as const;

export interface Sample {
  id: number;
  inputs: number[];
  targets: number[];
}
export interface NumericalFixture {
  seed: number;
  width: number;
  initialParameters: number[][];
  samples: Sample[];
  description: string;
  kind?: "dense" | "moe";
}
export interface ReferenceResult {
  gradients: number[][];
  parameters: number[][];
  firstMoments: number[][];
  secondMoments: number[][];
  lossBefore: number;
  lossAfter: number;
}
export interface NumericalRankSnapshot {
  rank: number;
  sampleIds: number[];
  parameterRanges: [number, number][];
  parameterIndices: number[][];
  optimizerIndices: number[][];
  synchronizedGradientIndices: number[][];
  parameters: number[][];
  gradients: (number[] | null)[];
  gathered: (number[] | null)[];
  synchronizedLayers: number[];
  updatedLayers: number[];
  parameterSynchronizedLayers: number[];
  localLoss: number | null;
}
export interface NumericalSnapshot {
  eventIndex: number;
  time: number;
  operation: string | null;
  rankStates: NumericalRankSnapshot[];
}
export interface NumericalResult {
  fixture: NumericalFixture;
  reference: ReferenceResult;
  localGradients: number[][][];
  synchronizedGradients: number[][][];
  parametersByRank: number[][][];
  firstMomentsByRank: number[][][];
  secondMomentsByRank: number[][][];
  reconstructedParameters: number[][];
  sampleIdsByRank: number[][];
  parameterRangesByRank: [number, number][][];
  parameterIndicesByRank: number[][][];
  optimizerIndicesByRank: number[][][];
  synchronizedGradientIndicesByRank: number[][][];
  lossBefore: number;
  lossAfter: number;
  maxGradientError: number;
  maxParameterError: number;
  maxReplicaError: number;
  tolerance: number;
  history: NumericalSnapshot[];
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function createFixture(
  seed: number,
  globalBatch: number,
  kind: "dense" | "moe" = "dense",
): NumericalFixture {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff)
    throw new Error("The fixture seed must be an unsigned 32-bit integer.");
  if (!Number.isInteger(globalBatch) || globalBatch < 1 || globalBatch > 128)
    throw new Error("The fixture batch must be an integer from 1 to 128.");
  const random = seededRandom(seed);
  const initialParameters = Array.from(
    { length: kind === "moe" ? EXPERT_COUNT : fixtureLayers },
    () =>
      Array.from(
        { length: fixtureWidth * fixtureWidth },
        (_, index) =>
          (Math.floor(index / fixtureWidth) === index % fixtureWidth
            ? 0.8
            : 0) +
          (random() - 0.5) * 0.1,
      ),
  );
  const teacher = Array.from(
    { length: fixtureWidth },
    () => 0.4 + random() * 0.2,
  );
  const samples = Array.from({ length: globalBatch }, (_, id) => {
    const inputs = Array.from({ length: fixtureWidth }, () => 0.25 + random());
    const targets = inputs.map(
      (input, channel) => input * teacher[channel] + (random() - 0.5) * 0.02,
    );
    return { id, inputs, targets };
  });
  return {
    seed,
    width: fixtureWidth,
    initialParameters,
    samples,
    kind,
    description:
      kind === "moe"
        ? "Eight independent 8×8 linear experts. Fixed top-1 routing selects (sample ID × 5 + seed) modulo 8. Loss averages all source tokens and output channels. JavaScript numbers do not emulate FP32 or BF16 rounding."
        : "Four sequential 8×8 dense linear layers, with input-major matrix entries. Loss is half squared error averaged over samples and output channels. JavaScript numbers do not emulate FP32 or BF16 rounding.",
  };
}

export function partitionSamples(samples: Sample[], ranks: number): Sample[][] {
  if (
    !Number.isInteger(ranks) ||
    ranks < 1 ||
    samples.length % ranks !== 0 ||
    samples.length === 0
  )
    throw new Error(
      "Equal nonempty local batches require globalBatch to be divisible by ranks.",
    );
  const localBatch = samples.length / ranks;
  return Array.from({ length: ranks }, (_, rank) =>
    samples.slice(rank * localBatch, (rank + 1) * localBatch),
  );
}

function fullForward(
  input: number[],
  weights: number[],
  width: number,
): number[] {
  return Array.from({ length: width }, (_, output) =>
    input.reduce(
      (sum, value, index) => sum + value * weights[index * width + output],
      0,
    ),
  );
}

export function lossFor(
  parameters: number[][],
  samples: Sample[],
  routes?: number[],
): number {
  const width = samples[0].inputs.length;
  let squaredError = 0;
  samples.forEach((sample, sampleIndex) => {
    let prediction = [...sample.inputs];
    const matrices = routes ? [parameters[routes[sampleIndex]]] : parameters;
    for (const matrix of matrices)
      prediction = fullForward(prediction, matrix, width);
    for (let channel = 0; channel < width; channel++)
      squaredError += (prediction[channel] - sample.targets[channel]) ** 2;
  });
  return squaredError / (2 * samples.length * width);
}

export function adamFirstStep(
  parameter: number,
  gradient: number,
  learningRate: number,
) {
  const { beta1, beta2, epsilon } = adamSettings;
  const firstMoment = (1 - beta1) * gradient;
  const secondMoment = (1 - beta2) * gradient * gradient;
  return {
    parameter:
      parameter -
      (learningRate * (firstMoment / (1 - beta1))) /
        (Math.sqrt(secondMoment / (1 - beta2)) + epsilon),
    firstMoment,
    secondMoment,
  };
}

export function referenceStep(
  fixture: NumericalFixture,
  learningRate: number,
): ReferenceResult {
  if (!Number.isFinite(learningRate) || learningRate <= 0)
    throw new Error("The learning rate must be positive.");
  const { initialParameters, samples, width } = fixture;
  if (initialParameters.some((matrix) => matrix.length !== width * width))
    throw new Error(
      "Each representative parameter group must be a square matrix.",
    );
  const gradients = initialParameters.map((matrix) => matrix.map(() => 0));
  const identity: number[] = Array.from(
    { length: width * width },
    (_, index) => (Math.floor(index / width) === index % width ? 1 : 0),
  );
  const multiply = (left: number[], right: number[]) =>
    Array.from({ length: width * width }, (_, index) => {
      const row = Math.floor(index / width),
        column = index % width;
      let sum = 0;
      for (let inner = 0; inner < width; inner++)
        sum += left[row * width + inner] * right[inner * width + column];
      return sum;
    });
  const suffix = initialParameters.map(() => identity);
  if (fixture.kind !== "moe")
    for (let layer = initialParameters.length - 2; layer >= 0; layer--)
      suffix[layer] = multiply(initialParameters[layer + 1], suffix[layer + 1]);
  const routes =
    fixture.kind === "moe"
      ? samples.map((sample) => expertForSample(sample.id, fixture.seed))
      : undefined;
  samples.forEach((sample, sampleIndex) => {
    const groupIds = routes
      ? [routes[sampleIndex]]
      : initialParameters.map((_, layer) => layer);
    const inputs: number[][] = [];
    let prediction = [...sample.inputs];
    for (const group of groupIds) {
      inputs.push(prediction);
      prediction = fullForward(prediction, initialParameters[group], width);
    }
    groupIds.forEach((group, position) => {
      for (let input = 0; input < width; input++)
        for (let output = 0; output < width; output++) {
          let residualDerivative = 0;
          for (let final = 0; final < width; final++)
            residualDerivative +=
              (prediction[final] - sample.targets[final]) *
              (routes ? identity : suffix[group])[output * width + final];
          gradients[group][input * width + output] +=
            (inputs[position][input] * residualDerivative) /
            (samples.length * width);
        }
    });
  });
  const updates = initialParameters.map((matrix, group) =>
    matrix.map((parameter, index) =>
      adamFirstStep(parameter, gradients[group][index], learningRate),
    ),
  );
  const parameters = updates.map((matrix) =>
    matrix.map((update) => update.parameter),
  );
  return {
    gradients,
    parameters,
    firstMoments: updates.map((matrix) =>
      matrix.map((update) => update.firstMoment),
    ),
    secondMoments: updates.map((matrix) =>
      matrix.map((update) => update.secondMoment),
    ),
    lossBefore: lossFor(initialParameters, samples, routes),
    lossAfter: lossFor(parameters, samples, routes),
  };
}

type Matrix = number[][];
interface RankState extends NumericalRankSnapshot {
  samples: Sample[];
  inputs: Map<string, Matrix>;
  outputs: Map<string, Matrix>;
  derivatives: Map<string, Matrix>;
  inputDerivatives: Map<string, Matrix>;
  expertSamples: Map<number, Sample[]>;
  sourceOutputs: Map<number, number[]>;
  returnedInputGradients: Map<number, number[]>;
  localGradients: (number[] | null)[];
  synchronizedGradients: (number[] | null)[];
  firstMoments: number[][];
  secondMoments: number[][];
}
const matrixCopy = (matrix: Matrix): Matrix => matrix.map((row) => [...row]);
const optionalMatrixCopy = (matrix: (number[] | null)[]) =>
  matrix.map((row) => (row === null ? null : [...row]));
const key = (layer: number, microbatch = 0) => `${layer}:${microbatch}`;
function required<T>(value: T | null | undefined, message: string): T {
  if (value === null || value === undefined) throw new Error(message);
  return value;
}
function snapshot(
  states: RankState[],
  event: SimulationEvent,
): NumericalSnapshot {
  return {
    eventIndex: event.index,
    time: event.time,
    operation: event.operation,
    rankStates: states.map((state) => ({
      rank: state.rank,
      sampleIds: [...state.sampleIds],
      parameterRanges: state.parameterRanges.map(([start, end]) => [
        start,
        end,
      ]),
      parameterIndices: matrixCopy(state.parameterIndices),
      optimizerIndices: matrixCopy(state.optimizerIndices),
      synchronizedGradientIndices: matrixCopy(
        state.synchronizedGradientIndices,
      ),
      parameters: matrixCopy(state.parameters),
      gradients: optionalMatrixCopy(state.gradients),
      gathered: optionalMatrixCopy(state.gathered),
      synchronizedLayers: [...state.synchronizedLayers],
      updatedLayers: [...state.updatedLayers],
      parameterSynchronizedLayers: [...state.parameterSynchronizedLayers],
      localLoss: state.localLoss,
    })),
  };
}
function maximumDifference(a: Matrix, b: Matrix): number {
  return Math.max(
    0,
    ...a.flatMap((row, layer) =>
      row.map((value, index) => Math.abs(value - b[layer][index])),
    ),
  );
}
function partialForward(
  inputs: Matrix,
  weights: number[],
  indices: number[],
  width: number,
): Matrix {
  return inputs.map((input) => {
    const output = Array<number>(width).fill(0);
    indices.forEach((index, owned) => {
      output[index % width] +=
        input[Math.floor(index / width)] * weights[owned];
    });
    return output;
  });
}
function partialBackward(
  inputs: Matrix,
  derivative: Matrix,
  weights: number[],
  indices: number[],
  width: number,
) {
  const gradient = indices.map((index) =>
    inputs.reduce(
      (sum, input, sample) =>
        sum +
        input[Math.floor(index / width)] * derivative[sample][index % width],
      0,
    ),
  );
  const inputDerivative = derivative.map((row) => {
    const output = Array<number>(width).fill(0);
    indices.forEach((index, owned) => {
      output[Math.floor(index / width)] += row[index % width] * weights[owned];
    });
    return output;
  });
  return { gradient, inputDerivative };
}
function sumMatrices(matrices: Matrix[]): Matrix {
  return matrices[0].map((row, sample) =>
    row.map((_, channel) =>
      matrices.reduce((sum, matrix) => sum + matrix[sample][channel], 0),
    ),
  );
}

export function executeNumerics(run: Simulation): NumericalResult {
  const { config } = run;
  const shardedDataParallel = isShardedDataParallel(config.strategy);
  const shardedReplicaOptimizer =
    config.strategy === "zero1" || config.strategy === "zero2";
  const replicatedParameters =
    config.strategy === "ddp" || shardedReplicaOptimizer;
  const fixture = createFixture(
    config.seed,
    config.globalBatch,
    config.strategy === "ep" ? "moe" : "dense",
  );
  const { width, initialParameters } = fixture;
  const groups = initialParameters.length;
  if (run.layers.length !== groups)
    throw new Error(
      "The virtual and representative parameter groups must match.",
    );
  const allIndices = Array.from({ length: width * width }, (_, index) => index);
  const localSamples =
    config.strategy === "tp" || config.strategy === "pp"
      ? Array.from({ length: config.ranks }, () => fixture.samples)
      : partitionSamples(fixture.samples, config.ranks);
  const states: RankState[] = localSamples.map((samples, rank) => {
    const indices = initialParameters.map((_, layer) =>
      parameterIndices(config.strategy, layer, rank, config.ranks, width),
    );
    const parameters = initialParameters.map((matrix, layer) =>
      indices[layer].map((index) => matrix[index]),
    );
    const optimizerIndices = shardedReplicaOptimizer
      ? initialParameters.map((matrix) => {
          const [start, end] = optimizerRange(config, matrix.length, rank);
          return allIndices.slice(start, end);
        })
      : matrixCopy(indices);
    const synchronizedGradientIndices =
      config.strategy === "zero2"
        ? matrixCopy(optimizerIndices)
        : matrixCopy(indices);
    return {
      rank,
      samples,
      sampleIds: samples.map((sample) => sample.id),
      parameterIndices: indices,
      optimizerIndices,
      synchronizedGradientIndices,
      parameterRanges: indices.map((owned) =>
        owned.length ? [Math.min(...owned), Math.max(...owned) + 1] : [0, 0],
      ),
      parameters,
      gradients: initialParameters.map(() => null),
      gathered: initialParameters.map(() => null),
      synchronizedLayers: [],
      updatedLayers: [],
      parameterSynchronizedLayers: [],
      localLoss: null,
      inputs: new Map(),
      outputs: new Map(),
      derivatives: new Map(),
      inputDerivatives: new Map(),
      expertSamples: new Map(),
      sourceOutputs: new Map(),
      returnedInputGradients: new Map(),
      localGradients: initialParameters.map(() => null),
      synchronizedGradients: initialParameters.map(() => null),
      firstMoments: optimizerIndices.map((row) => row.map(() => 0)),
      secondMoments: optimizerIndices.map((row) => row.map(() => 0)),
    };
  });
  const operations = new Map(
    run.operations.map((operation) => [operation.id, operation]),
  );
  const capturedParameters = new Map<
    string,
    Map<number, Map<number, number[]>>
  >();
  const started = new Set<string>(),
    finished = new Set<string>();
  const history: NumericalSnapshot[] = [];
  const samplesFor = (
    state: RankState,
    operation: Operation,
    layer: number,
  ): Sample[] => {
    if (config.strategy === "ep")
      return required(
        state.expertSamples.get(layer),
        "Expert inputs have not been dispatched.",
      );
    if (config.strategy !== "pp") return state.samples;
    const microbatch = required(
      operation.microbatch,
      "Pipeline compute needs a microbatch.",
    );
    const size = config.globalBatch / config.microbatches;
    return state.samples.slice(microbatch * size, (microbatch + 1) * size);
  };
  const computationIndices = (state: RankState, layer: number) =>
    shardedDataParallel ? allIndices : state.parameterIndices[layer];
  const parametersFor = (state: RankState, layer: number) =>
    shardedDataParallel
      ? required(
          state.gathered[layer],
          `Rank ${state.rank} consumed layer ${layer} before all-gather completed.`,
        )
      : state.parameters[layer];
  const forwardInputs = (
    state: RankState,
    operation: Operation,
    layer: number,
  ): Matrix =>
    config.strategy === "ep" || layer === 0
      ? samplesFor(state, operation, layer).map((sample) => [...sample.inputs])
      : required(
          state.outputs.get(key(layer - 1, operation.microbatch)),
          "Previous forward output or pipeline receive is missing.",
        );
  const derivativeFor = (
    state: RankState,
    operation: Operation,
    layer: number,
  ): Matrix => {
    if (config.strategy === "ep" || layer !== groups - 1)
      return required(
        state.derivatives.get(key(layer, operation.microbatch)),
        "The upstream gradient or backward receive is missing.",
      );
    const samples = samplesFor(state, operation, layer);
    const output = required(
      state.outputs.get(key(layer, operation.microbatch)),
      "Final forward output is missing.",
    );
    const count =
      isZeroStrategy(config.strategy) || shardedDataParallel
        ? state.samples.length
        : config.globalBatch;
    return output.map((row, sample) =>
      row.map(
        (value, channel) =>
          (value - samples[sample].targets[channel]) / (count * width),
      ),
    );
  };
  const syncOwnedGradient = (
    state: RankState,
    layer: number,
    gradient: number[],
  ) => {
    state.synchronizedGradients[layer] = [...gradient];
    state.gradients[layer] = [...gradient];
    if (!state.synchronizedLayers.includes(layer))
      state.synchronizedLayers.push(layer);
  };
  const outputLoss = (outputs: Matrix, samples: Sample[]) =>
    outputs.reduce(
      (sum, row, sample) =>
        sum +
        row.reduce(
          (error, value, channel) =>
            error + (value - samples[sample].targets[channel]) ** 2,
          0,
        ),
      0,
    ) /
    (2 * width);
  function start(operation: Operation) {
    if (started.has(operation.id))
      throw new Error(`Operation ${operation.id} started twice.`);
    for (const dependency of operation.prerequisites)
      if (
        !(dependency.phase === "start" ? started : finished).has(
          dependency.operation,
        )
      )
        throw new Error(
          `Operation ${operation.id} started before prerequisite ${dependency.operation} ${dependency.phase}.`,
        );
    if (operation.kind === "forward" || operation.kind === "backward") {
      const captured = new Map<number, Map<number, number[]>>();
      for (const rank of operation.ranks) {
        const state = states[rank],
          layers = new Map<number, number[]>();
        for (const layer of operation.layers) {
          if (!state.parameterIndices[layer].length && !shardedDataParallel)
            throw new Error("Compute rank does not own the parameter group.");
          layers.set(layer, [...parametersFor(state, layer)]);
          if (operation.kind === "forward")
            forwardInputs(state, operation, layer);
          else {
            required(
              state.inputs.get(key(layer, operation.microbatch)),
              "Backward saved input is missing.",
            );
            derivativeFor(state, operation, layer);
          }
        }
        captured.set(rank, layers);
      }
      capturedParameters.set(operation.id, captured);
    }
    if (operation.kind === "adam")
      for (const rank of operation.ranks)
        for (const layer of operation.layers)
          required(
            states[rank].synchronizedGradients[layer],
            "Adam started before gradient synchronization or accumulation.",
          );
    if (operation.purpose === "parameter-sync") {
      if (operation.kind !== "all-gather" || !shardedReplicaOptimizer)
        throw new Error(
          "Parameter synchronization requires a ZeRO-1 or ZeRO-2 all-gather.",
        );
      if (
        new Set(operation.ranks).size !== states.length ||
        operation.ranks.length !== states.length
      )
        throw new Error(
          "Parameter synchronization must include every replica rank once.",
        );
      for (const state of states)
        for (const layer of operation.layers) {
          if (!state.updatedLayers.includes(layer))
            throw new Error(
              "Parameter synchronization started before every owner completed Adam.",
            );
          if (state.parameterSynchronizedLayers.includes(layer))
            throw new Error("A parameter layer was synchronized twice.");
        }
    }
    started.add(operation.id);
  }
  function finish(operation: Operation) {
    if (!started.has(operation.id) || finished.has(operation.id))
      throw new Error(`Invalid completion for operation ${operation.id}.`);
    switch (operation.kind) {
      case "all-gather":
        for (const layer of operation.layers) {
          const full = Array<number>(width * width).fill(0),
            owners = Array<number>(width * width).fill(0);
          const parameterSync = operation.purpose === "parameter-sync";
          for (const state of states)
            (parameterSync
              ? state.optimizerIndices[layer]
              : state.parameterIndices[layer]
            ).forEach((index) => {
              const local = state.parameterIndices[layer].indexOf(index);
              full[index] = state.parameters[layer][local];
              owners[index]++;
            });
          if (owners.some((count) => count !== 1))
            throw new Error(
              "All-gather requires exactly one owner per parameter.",
            );
          for (const rank of operation.ranks) {
            if (parameterSync) {
              states[rank].parameters[layer] = [...full];
              states[rank].parameterSynchronizedLayers.push(layer);
            } else states[rank].gathered[layer] = [...full];
          }
        }
        break;
      case "forward":
        for (const rank of operation.ranks) {
          const state = states[rank];
          for (const layer of operation.layers) {
            const parameters = required(
              capturedParameters.get(operation.id)?.get(rank)?.get(layer),
              "Forward parameters were not captured at start.",
            );
            const inputs = forwardInputs(state, operation, layer);
            state.inputs.set(
              key(layer, operation.microbatch),
              matrixCopy(inputs),
            );
            const outputs = partialForward(
              inputs,
              parameters,
              computationIndices(state, layer),
              width,
            );
            state.outputs.set(key(layer, operation.microbatch), outputs);
            if (
              layer === groups - 1 &&
              config.strategy !== "tp" &&
              config.strategy !== "ep"
            ) {
              const samples = samplesFor(state, operation, layer);
              const count =
                config.strategy === "pp"
                  ? config.globalBatch
                  : state.samples.length;
              const loss = outputLoss(outputs, samples) / count;
              state.localLoss =
                config.strategy === "pp" ? (state.localLoss ?? 0) + loss : loss;
            }
            if (shardedDataParallel && config.reshardAfterForward)
              state.gathered[layer] = null;
          }
        }
        break;
      case "backward":
        for (const rank of operation.ranks) {
          const state = states[rank];
          for (const layer of operation.layers) {
            const parameters = required(
              capturedParameters.get(operation.id)?.get(rank)?.get(layer),
              "Backward parameters were not captured at start.",
            );
            const inputs = required(
              state.inputs.get(key(layer, operation.microbatch)),
              "Backward saved input is missing.",
            );
            const { gradient, inputDerivative } = partialBackward(
              inputs,
              derivativeFor(state, operation, layer),
              parameters,
              computationIndices(state, layer),
              width,
            );
            const accumulated =
              config.strategy === "pp" && state.localGradients[layer]
                ? gradient.map(
                    (value, index) =>
                      value + state.localGradients[layer]![index],
                  )
                : gradient;
            state.localGradients[layer] = accumulated;
            state.gradients[layer] = [...accumulated];
            if (
              config.strategy === "tp" ||
              config.strategy === "ep" ||
              (config.strategy === "pp" && operation.microbatch === 0)
            )
              syncOwnedGradient(state, layer, accumulated);
            state.inputDerivatives.set(
              key(layer, operation.microbatch),
              inputDerivative,
            );
            if (
              layer > 0 &&
              config.strategy !== "ep" &&
              !(config.strategy === "tp" && layer % 2 === 0) &&
              !(
                config.strategy === "pp" &&
                stageOwner(layer - 1, config.ranks) !== rank
              )
            )
              state.derivatives.set(
                key(layer - 1, operation.microbatch),
                matrixCopy(inputDerivative),
              );
            state.inputs.delete(key(layer, operation.microbatch));
            state.outputs.delete(key(layer, operation.microbatch));
            state.derivatives.delete(key(layer, operation.microbatch));
            if (shardedDataParallel) state.gathered[layer] = null;
          }
        }
        break;
      case "all-reduce":
      case "reduce-scatter":
        for (const layer of operation.layers) {
          if (config.strategy === "tp") {
            const forward = operation.phaseTag === "tp-forward-sum";
            if (!forward && operation.phaseTag !== "tp-backward-sum")
              throw new Error(
                "TP reductions need an activation or input-gradient purpose.",
              );
            if (operation.reduction !== "sum")
              throw new Error("Tensor-parallel contributions require SUM.");
            const total = sumMatrices(
              operation.ranks.map((rank) =>
                required(
                  (forward
                    ? states[rank].outputs
                    : states[rank].inputDerivatives
                  ).get(key(layer)),
                  "The partial tensor is not ready for reduction.",
                ),
              ),
            );
            for (const rank of operation.ranks) {
              if (forward) {
                states[rank].outputs.set(key(layer), matrixCopy(total));
                if (layer === groups - 1)
                  states[rank].localLoss =
                    outputLoss(total, fixture.samples) / config.globalBatch;
              } else {
                states[rank].inputDerivatives.set(
                  key(layer),
                  matrixCopy(total),
                );
                if (layer > 0)
                  states[rank].derivatives.set(
                    key(layer - 1),
                    matrixCopy(total),
                  );
              }
            }
          } else {
            const gradients = states.map((state) =>
              required(
                state.localGradients[layer],
                "A gradient is not ready for synchronization.",
              ),
            );
            const average = allIndices.map(
              (index) =>
                gradients.reduce((sum, gradient) => sum + gradient[index], 0) /
                states.length,
            );
            for (const rank of operation.ranks)
              syncOwnedGradient(
                states[rank],
                layer,
                states[rank].synchronizedGradientIndices[layer].map(
                  (index) => average[index],
                ),
              );
          }
        }
        break;
      case "send-recv": {
        const layer = operation.layers[0],
          microbatch = required(
            operation.microbatch,
            "Pipeline transfers need a microbatch.",
          );
        if (operation.phaseTag === "pp-forward-transfer") {
          const source = states[stageOwner(layer, config.ranks)],
            destination = states[stageOwner(layer + 1, config.ranks)];
          destination.outputs.set(
            key(layer, microbatch),
            matrixCopy(
              required(
                source.outputs.get(key(layer, microbatch)),
                "Pipeline sender has no output.",
              ),
            ),
          );
        } else if (operation.phaseTag === "pp-backward-transfer") {
          const source = states[stageOwner(layer, config.ranks)],
            destination = states[stageOwner(layer - 1, config.ranks)];
          destination.derivatives.set(
            key(layer - 1, microbatch),
            matrixCopy(
              required(
                source.inputDerivatives.get(key(layer, microbatch)),
                "Pipeline sender has no input derivative.",
              ),
            ),
          );
        } else throw new Error("A pipeline transfer needs a direction.");
        break;
      }
      case "all-to-all":
        if (operation.phaseTag === "ep-dispatch") {
          for (let expert = 0; expert < groups; expert++)
            states[expertOwner(expert, config.ranks)].expertSamples.set(
              expert,
              [],
            );
          for (const source of states)
            for (const sample of source.samples) {
              const expert = expertForSample(sample.id, config.seed);
              states[expertOwner(expert, config.ranks)].expertSamples
                .get(expert)!
                .push(sample);
            }
        } else if (operation.phaseTag === "ep-combine") {
          for (let expert = 0; expert < groups; expert++) {
            const owner = states[expertOwner(expert, config.ranks)];
            const samples = required(
              owner.expertSamples.get(expert),
              "Expert samples are missing.",
            );
            const outputs = required(
              owner.outputs.get(key(expert)),
              "Expert output is not ready for combine.",
            );
            samples.forEach((sample, index) =>
              states[
                Math.floor((sample.id * config.ranks) / config.globalBatch)
              ].sourceOutputs.set(sample.id, [...outputs[index]]),
            );
          }
          for (const source of states)
            source.localLoss =
              outputLoss(
                source.samples.map((sample) =>
                  required(
                    source.sourceOutputs.get(sample.id),
                    "Combine did not return a token.",
                  ),
                ),
                source.samples,
              ) / source.samples.length;
        } else if (operation.phaseTag === "ep-backward-dispatch") {
          for (let expert = 0; expert < groups; expert++) {
            const owner = states[expertOwner(expert, config.ranks)];
            const samples = required(
              owner.expertSamples.get(expert),
              "Expert samples are missing.",
            );
            owner.derivatives.set(
              key(expert),
              samples.map((sample) => {
                const source =
                  states[
                    Math.floor((sample.id * config.ranks) / config.globalBatch)
                  ];
                return required(
                  source.sourceOutputs.get(sample.id),
                  "Loss derivative needs the combined output.",
                ).map(
                  (value, channel) =>
                    (value - sample.targets[channel]) /
                    (config.globalBatch * width),
                );
              }),
            );
          }
        } else if (operation.phaseTag === "ep-backward-combine") {
          for (let expert = 0; expert < groups; expert++) {
            const owner = states[expertOwner(expert, config.ranks)];
            const derivatives = required(
              owner.inputDerivatives.get(key(expert)),
              "Expert input derivatives are missing.",
            );
            required(
              owner.expertSamples.get(expert),
              "Expert samples are missing.",
            ).forEach((sample, index) =>
              states[
                Math.floor((sample.id * config.ranks) / config.globalBatch)
              ].returnedInputGradients.set(sample.id, [...derivatives[index]]),
            );
          }
        } else throw new Error("Expert communication needs a routing phase.");
        break;
      case "adam":
        for (const rank of operation.ranks) {
          const state = states[rank];
          for (const layer of operation.layers) {
            if (state.updatedLayers.includes(layer))
              throw new Error(`Rank ${rank} updated layer ${layer} twice.`);
            const gradient = required(
              state.synchronizedGradients[layer],
              "Synchronized gradients are missing.",
            );
            const updates = state.optimizerIndices[layer].map((index) => {
              const parameterIndex =
                state.parameterIndices[layer].indexOf(index);
              const gradientIndex =
                state.synchronizedGradientIndices[layer].indexOf(index);
              if (parameterIndex < 0 || gradientIndex < 0)
                throw new Error(
                  "Adam requires a local parameter and synchronized gradient for every optimizer-owned index.",
                );
              const update = adamFirstStep(
                state.parameters[layer][parameterIndex],
                gradient[gradientIndex],
                config.learningRate,
              );
              state.parameters[layer][parameterIndex] = update.parameter;
              return update;
            });
            state.firstMoments[layer] = updates.map(
              (update) => update.firstMoment,
            );
            state.secondMoments[layer] = updates.map(
              (update) => update.secondMoment,
            );
            state.gradients[layer] = null;
            state.updatedLayers.push(layer);
          }
        }
        break;
    }
    capturedParameters.delete(operation.id);
    finished.add(operation.id);
  }
  for (const event of run.events) {
    if (event.phase === "initialize") {
      history.push(snapshot(states, event));
      continue;
    }
    const operation = required(
      operations.get(
        required(event.operation, "A noninitial event needs an operation."),
      ),
      "The numerical event names an unknown operation.",
    );
    if (event.phase === "start") start(operation);
    else {
      finish(operation);
      history.push(snapshot(states, event));
    }
  }
  for (const state of states) {
    const owned = state.optimizerIndices.flatMap((indices, layer) =>
      indices.length ? [layer] : [],
    );
    if (
      owned.some((layer) => !state.updatedLayers.includes(layer)) ||
      state.updatedLayers.length !== owned.length
    )
      throw new Error(
        `Rank ${state.rank} did not update every owned group exactly once.`,
      );
    if (
      shardedReplicaOptimizer &&
      owned.some((layer) => !state.parameterSynchronizedLayers.includes(layer))
    )
      throw new Error(
        "Updated parameter replicas are missing their final parameter synchronization.",
      );
    if (
      config.strategy === "ep" &&
      state.returnedInputGradients.size !== state.samples.length
    )
      throw new Error(
        "Backward combine did not return every source token's input derivative.",
      );
  }
  const finalGradients = (state: RankState, gradients: (number[] | null)[]) =>
    gradients.map((gradient, layer) => {
      if (gradient !== null) return [...gradient];
      if (state.parameterIndices[layer].length === 0) return [];
      throw new Error(
        "An owned parameter group is missing its completed gradient.",
      );
    });
  const localGradients = states.map((state) =>
    finalGradients(state, state.localGradients),
  );
  const synchronizedGradients = states.map((state) =>
    finalGradients(state, state.synchronizedGradients),
  );
  const parametersByRank = states.map((state) => matrixCopy(state.parameters));
  const reconstructedParameters = initialParameters.map((_, layer) => {
    const output = Array<number>(width * width).fill(0),
      owners = Array<number>(width * width).fill(0);
    for (const state of replicatedParameters ? [states[0]] : states)
      state.parameterIndices[layer].forEach((index, local) => {
        output[index] = state.parameters[layer][local];
        owners[index]++;
      });
    if (owners.some((count) => count !== 1))
      throw new Error(
        "Parameter reconstruction requires exactly one owner per matrix entry.",
      );
    return output;
  });
  const reference = referenceStep(fixture, config.learningRate);
  const expectedOwned = (matrix: Matrix, indices: number[][]) =>
    matrix.map((row, layer) => indices[layer].map((index) => row[index]));
  const routes =
    config.strategy === "ep"
      ? fixture.samples.map((sample) => expertForSample(sample.id, config.seed))
      : undefined;
  const losses = states
    .map((state) => state.localLoss)
    .filter((loss): loss is number => loss !== null);
  if (!losses.length)
    throw new Error("The forward pass did not calculate loss.");
  return {
    fixture,
    reference,
    localGradients,
    synchronizedGradients,
    parametersByRank,
    firstMomentsByRank: states.map((state) => matrixCopy(state.firstMoments)),
    secondMomentsByRank: states.map((state) => matrixCopy(state.secondMoments)),
    reconstructedParameters,
    sampleIdsByRank: states.map((state) => [...state.sampleIds]),
    parameterRangesByRank: states.map((state) =>
      state.parameterRanges.map(([start, end]) => [start, end]),
    ),
    parameterIndicesByRank: states.map((state) =>
      matrixCopy(state.parameterIndices),
    ),
    optimizerIndicesByRank: states.map((state) =>
      matrixCopy(state.optimizerIndices),
    ),
    synchronizedGradientIndicesByRank: states.map((state) =>
      matrixCopy(state.synchronizedGradientIndices),
    ),
    lossBefore: losses.reduce((sum, loss) => sum + loss, 0) / losses.length,
    lossAfter: lossFor(reconstructedParameters, fixture.samples, routes),
    maxGradientError: Math.max(
      ...states.map((state, rank) =>
        maximumDifference(
          synchronizedGradients[rank],
          expectedOwned(reference.gradients, state.synchronizedGradientIndices),
        ),
      ),
    ),
    maxParameterError: Math.max(
      ...states.map((state, rank) =>
        maximumDifference(
          parametersByRank[rank],
          expectedOwned(reference.parameters, state.parameterIndices),
        ),
      ),
    ),
    maxReplicaError: replicatedParameters
      ? Math.max(
          ...parametersByRank.map((parameters) =>
            maximumDifference(parameters, parametersByRank[0]),
          ),
        )
      : 0,
    tolerance: numericalTolerance,
    history,
  };
}

export function getNumericalSnapshot(
  result: NumericalResult,
  eventIndex: number,
): NumericalSnapshot {
  let low = 0,
    high = result.history.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (result.history[middle].eventIndex <= eventIndex) low = middle;
    else high = middle - 1;
  }
  return result.history[low];
}
