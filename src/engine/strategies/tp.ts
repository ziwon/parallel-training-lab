import type { Category, Config, Layer } from "../types";
import { TOKENS_PER_SAMPLE } from "../model";
import { bufferId, createPlan, end } from "./plan";

export function tpPlan(config: Config, layers: Layer[]) {
  const { plan, ranks, allocate, operation } = createPlan(config, layers);
  const payload = (width: number) =>
    config.globalBatch * width * TOKENS_PER_SAMPLE * config.modelScale * 4;
  const buffer = (
    kind: string,
    layer: Layer,
    rank: number,
    bytes: number,
    category: Category,
    label: string,
  ) =>
    allocate({
      id: bufferId(kind, layer.id, rank),
      rank,
      layer: layer.id,
      bytes,
      category,
      ownership: "local",
      label,
      range: [0, bytes / 4],
    });
  const output = (layer: number, rank: number) =>
    bufferId(layer % 2 === 1 ? "ysum" : "y", layer, rank);
  const derivative = (layer: number, rank: number) =>
    bufferId(layer % 2 === 0 ? "dxsum" : "dx", layer, rank);

  for (const layer of layers) {
    const column = layer.id % 2 === 0;
    for (const rank of ranks) {
      const input = layer.id > 0 ? output(layer.id - 1, rank) : null;
      const a = buffer(
        "a",
        layer,
        rank,
        payload(layer.inputWidth) / (column ? 1 : config.ranks),
        "activations",
        "Saved tensor-parallel layer input",
      );
      const y = buffer(
        "y",
        layer,
        rank,
        payload(layer.outputWidth) / (column ? config.ranks : 1),
        "temporary",
        column ? "Complete output column slice" : "Partial row-parallel output",
      );
      operation(
        "forward",
        bufferId("f", layer.id, rank),
        [layer.id],
        [rank],
        layer.id > 0 && !column
          ? []
          : layer.id > 0
            ? [end(`tp:ar:f:${layer.id - 1}`)]
            : [],
        [bufferId("p", layer.id, rank), ...(input ? [input] : [])],
        [a, y],
        input ? [input] : [],
        {
          elements: layer.elements / config.ranks,
          samples: config.globalBatch,
        },
      );
    }
    if (!column) {
      const inputs = ranks.map((rank) => bufferId("y", layer.id, rank));
      const outputs = ranks.map((rank) =>
        buffer(
          "ysum",
          layer,
          rank,
          payload(layer.outputWidth),
          "temporary",
          "Complete output after SUM",
        ),
      );
      operation(
        "all-reduce",
        `tp:ar:f:${layer.id}`,
        [layer.id],
        ranks,
        ranks.map((rank) => end(bufferId("f", layer.id, rank))),
        inputs,
        outputs,
        inputs,
        {
          logicalBytes: payload(layer.outputWidth),
          phaseTag: "tp-forward-sum",
          reduction: "sum",
        },
      );
    }
  }
  for (const layer of [...layers].reverse()) {
    const column = layer.id % 2 === 0;
    for (const rank of ranks) {
      const upstream =
        layer.id === layers.length - 1
          ? output(layer.id, rank)
          : derivative(layer.id + 1, rank);
      const a = bufferId("a", layer.id, rank);
      const g = buffer(
        "g",
        layer,
        rank,
        (layer.elements * 4) / config.ranks,
        "gradients",
        "Owned tensor-parallel parameter gradient",
      );
      const dx = buffer(
        "dx",
        layer,
        rank,
        payload(layer.inputWidth) / (column ? 1 : config.ranks),
        "temporary",
        column
          ? "Partial input gradient before SUM"
          : "Complete input-gradient slice",
      );
      const prerequisite =
        layer.id === layers.length - 1
          ? `tp:ar:f:${layer.id}`
          : (layer.id + 1) % 2 === 0
            ? `tp:ar:b:${layer.id + 1}`
            : bufferId("b", layer.id + 1, rank);
      operation(
        "backward",
        bufferId("b", layer.id, rank),
        [layer.id],
        [rank],
        [end(prerequisite)],
        [bufferId("p", layer.id, rank), a, upstream],
        [g, dx],
        [a, upstream],
        {
          elements: layer.elements / config.ranks,
          samples: config.globalBatch,
        },
      );
    }
    if (column) {
      const inputs = ranks.map((rank) => bufferId("dx", layer.id, rank));
      const outputs = ranks.map((rank) =>
        buffer(
          "dxsum",
          layer,
          rank,
          payload(layer.inputWidth),
          "temporary",
          "Complete input gradient after SUM",
        ),
      );
      operation(
        "all-reduce",
        `tp:ar:b:${layer.id}`,
        [layer.id],
        ranks,
        ranks.map((rank) => end(bufferId("b", layer.id, rank))),
        inputs,
        outputs,
        [...inputs, ...(layer.id === 0 ? outputs : [])],
        {
          logicalBytes: payload(layer.inputWidth),
          phaseTag: "tp-backward-sum",
          reduction: "sum",
        },
      );
    }
  }
  for (const rank of ranks) {
    const gradients = layers.map((layer) => bufferId("g", layer.id, rank));
    operation(
      "adam",
      `adam:${rank}`,
      layers.map((layer) => layer.id),
      [rank],
      [end("tp:ar:b:0")],
      layers.flatMap((layer) => [
        bufferId("p", layer.id, rank),
        bufferId("m", layer.id, rank),
        bufferId("g", layer.id, rank),
      ]),
      [],
      gradients,
    );
  }
  return plan;
}
