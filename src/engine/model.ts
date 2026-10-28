import type { Config, Layer } from "./types";
import { engineMessages } from "./messages";

export const TOKENS_PER_SAMPLE = 128;
export const BYTES_PER_ELEMENT = 4;
export const MODEL_ASSUMPTIONS = engineMessages.modelAssumptions;

export function modelFor(
  config: Pick<Config, "modelScale"> & Partial<Pick<Config, "strategy">>,
): Layer[] {
  if (config.strategy === "ep")
    return Array.from({ length: 8 }, (_, id) => ({
      id,
      name: `Expert ${id}`,
      inputWidth: 1024,
      outputWidth: 1024,
      elements: 1024 * 1024 * config.modelScale,
      activationElementsPerSample: 1024 * TOKENS_PER_SAMPLE * config.modelScale,
    }));
  const widths = [1024, 2048, 4096, 2048, 1024];
  const names = engineMessages.layers;
  return names.map((name, id) => ({
    id,
    name,
    inputWidth: widths[id],
    outputWidth: widths[id + 1],
    elements: widths[id] * widths[id + 1] * config.modelScale,
    activationElementsPerSample:
      widths[id] * TOKENS_PER_SAMPLE * config.modelScale,
  }));
}

export function validateLayerSharding(layers: Layer[], ranks: number) {
  for (const layer of layers) {
    if (
      !Number.isSafeInteger(layer.elements) ||
      layer.elements <= 0 ||
      layer.elements % ranks !== 0
    ) {
      throw new Error(
        `Layer ${layer.id} must contain a whole number of parameter elements per rank.`,
      );
    }
  }
}
