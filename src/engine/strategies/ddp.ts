import { engineMessages } from "../messages";
import type { Config, Layer } from "../types";
import { bufferId, createPlan, end } from "./plan";

export function ddpPlan(config: Config, layers: Layer[]) {
  const { plan, ranks, allocate, operation } = createPlan(config, layers);
  const localBatch = config.globalBatch / config.ranks;
  for (const layer of layers)
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
      operation(
        "forward",
        bufferId("f", layer.id, rank),
        [layer.id],
        [rank],
        [],
        [bufferId("p", layer.id, rank)],
        [activation],
        [],
      );
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
        label: engineMessages.allocations.aliasedGradient,
        bytes: layer.elements * 4,
        range: [0, layer.elements],
      });
      operation(
        "backward",
        bufferId("b", layer.id, rank),
        [layer.id],
        [rank],
        [],
        [bufferId("p", layer.id, rank), bufferId("a", layer.id, rank)],
        [gradient],
        [bufferId("a", layer.id, rank)],
      );
    }
  const buckets: string[] = [];
  for (let offset = 0; offset < reverse.length; offset += config.bucketLayers) {
    const group = reverse
      .slice(offset, offset + config.bucketLayers)
      .map((layer) => layer.id);
    const id = `ar:${buckets.length}`;
    const prerequisites = group.flatMap((layer) =>
      ranks.map((rank) => end(bufferId("b", layer, rank))),
    );
    if (!config.overlap)
      prerequisites.push(...ranks.map((rank) => end(bufferId("b", 0, rank))));
    operation(
      "all-reduce",
      id,
      group,
      ranks,
      prerequisites,
      group.flatMap((layer) => ranks.map((rank) => bufferId("g", layer, rank))),
      [],
      [],
    );
    buckets.push(id);
  }
  for (const rank of ranks) {
    const gradients = layers.map((layer) => bufferId("g", layer.id, rank));
    operation(
      "adam",
      `adam:${rank}`,
      layers.map((layer) => layer.id),
      [rank],
      buckets.map(end),
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
