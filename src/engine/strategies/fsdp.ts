import { engineMessages } from "../messages";
import type { Config, Dependency, Layer } from "../types";
import { shardRange } from "../ownership";
import { bufferId, createPlan, end, start } from "./plan";

export function fsdpPlan(config: Config, layers: Layer[]) {
  const { plan, ranks, allocate, operation } = createPlan(config, layers);
  const localBatch = config.globalBatch / config.ranks;
  const prefetch = config.reshardAfterForward && config.prefetch === 1;
  const gathered = (phase: "f" | "b", layer: number, rank: number) =>
    bufferId(`pg${phase}`, layer, rank);

  function gather(phase: "f" | "b", layer: Layer, prerequisites: Dependency[]) {
    const outputs = ranks.map((rank) =>
      allocate({
        id: gathered(phase, layer.id, rank),
        rank,
        layer: layer.id,
        category: "temporary",
        ownership: "gathered",
        label: engineMessages.allocations.gathered[phase],
        bytes: layer.elements * 4,
        range: [0, layer.elements],
      }),
    );
    return operation(
      "all-gather",
      `ag:${phase}:${layer.id}`,
      [layer.id],
      ranks,
      prerequisites,
      ranks.map((rank) => bufferId("p", layer.id, rank)),
      outputs,
      [],
    );
  }
  function scatter(layer: Layer) {
    const outputs = ranks.map((rank) =>
      allocate({
        id: bufferId("gs", layer.id, rank),
        rank,
        layer: layer.id,
        category: "gradients",
        ownership: "shard",
        label: engineMessages.allocations.gradientShard,
        bytes: (layer.elements * 4) / config.ranks,
        range: shardRange(layer.elements, config.ranks, rank),
      }),
    );
    const inputs = ranks.map((rank) => bufferId("g", layer.id, rank));
    return operation(
      "reduce-scatter",
      `rs:${layer.id}`,
      [layer.id],
      ranks,
      ranks.map((rank) => end(bufferId("b", layer.id, rank))),
      inputs,
      outputs,
      inputs,
    );
  }

  for (const layer of layers) {
    gather(
      "f",
      layer,
      layer.id === 0
        ? []
        : ranks.map((rank) => end(bufferId("f", layer.id - 1, rank))),
    );
    for (const rank of ranks) {
      const activation = allocate({
        id: bufferId("a", layer.id, rank),
        rank,
        layer: layer.id,
        category: "activations",
        ownership: "local",
        label: engineMessages.allocations.savedInput,
        bytes: localBatch * layer.activationElementsPerSample * 4,
        range: [0, localBatch * layer.activationElementsPerSample],
      });
      const parameters = gathered("f", layer.id, rank);
      operation(
        "forward",
        bufferId("f", layer.id, rank),
        [layer.id],
        [rank],
        [end(`ag:f:${layer.id}`)],
        [parameters],
        [activation],
        config.reshardAfterForward ? [parameters] : [],
      );
    }
  }

  const reverse = [...layers].reverse();
  for (const layer of reverse)
    for (const rank of ranks) {
      const gradient = allocate({
        id: bufferId("g", layer.id, rank),
        rank,
        layer: layer.id,
        category: "gradients",
        ownership: "local",
        label: engineMessages.allocations.fullGradient,
        bytes: layer.elements * 4,
        range: [0, layer.elements],
      });
      const parameters = gathered(
        config.reshardAfterForward ? "b" : "f",
        layer.id,
        rank,
      );
      const prerequisites = config.reshardAfterForward
        ? [end(`ag:b:${layer.id}`)]
        : [];
      operation(
        "backward",
        bufferId("b", layer.id, rank),
        [layer.id],
        [rank],
        prerequisites,
        [parameters, bufferId("a", layer.id, rank)],
        [gradient],
        [parameters, bufferId("a", layer.id, rank)],
      );
    }

  if (!config.reshardAfterForward) {
    for (const layer of reverse) scatter(layer);
  } else if (!prefetch) {
    for (const layer of reverse) {
      const previous = layer.id === layers.length - 1 ? "f" : "b";
      const previousLayer = previous === "f" ? layer.id : layer.id + 1;
      gather(
        "b",
        layer,
        ranks.map((rank) => end(bufferId(previous, previousLayer, rank))),
      );
      scatter(layer);
    }
  } else {
    gather(
      "b",
      reverse[0],
      ranks.map((rank) => end(bufferId("f", reverse[0].id, rank))),
    );
    for (let index = 0; index < reverse.length; index++) {
      const current = reverse[index];
      const next = reverse[index + 1];
      if (next) {
        const prerequisites = ranks.map((rank) =>
          start(bufferId("b", current.id, rank)),
        );
        if (index > 0)
          prerequisites.push(
            ...ranks.map((rank) =>
              end(bufferId("b", reverse[index - 1].id, rank)),
            ),
          );
        gather("b", next, prerequisites);
      }
      scatter(current);
    }
  }
  for (const rank of ranks) {
    const gradients = layers.map((layer) => bufferId("gs", layer.id, rank));
    operation(
      "adam",
      `adam:${rank}`,
      layers.map((layer) => layer.id),
      [rank],
      layers.map((layer) => end(`rs:${layer.id}`)),
      layers.flatMap((layer) => [
        bufferId("p", layer.id, rank),
        bufferId("m", layer.id, rank),
        bufferId("gs", layer.id, rank),
      ]),
      [],
      gradients,
    );
  }
  return plan;
}
