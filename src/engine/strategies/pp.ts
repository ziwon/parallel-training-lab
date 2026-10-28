import type { Category, Config, Layer } from "../types";
import { TOKENS_PER_SAMPLE } from "../model";
import { stageOwner } from "../workload";
import { bufferId, createPlan, end } from "./plan";

export function ppPlan(config: Config, layers: Layer[]) {
  const { plan, ranks, allocate, operation } = createPlan(config, layers);
  const batch = config.globalBatch / config.microbatches;
  const owner = (layer: number) => stageOwner(layer, config.ranks);
  const id = (kind: string, layer: number, mb: number) =>
    `${kind}:${layer}:${owner(layer)}:${mb}`;
  const payload = (width: number) =>
    batch * width * TOKENS_PER_SAMPLE * config.modelScale * 4;
  const buffer = (
    name: string,
    layer: number,
    rank: number,
    bytes: number,
    category: Category,
    label: string,
  ) =>
    allocate({
      id: name,
      layer,
      rank,
      bytes,
      category,
      ownership: "local",
      label,
      range: [0, bytes / 4],
    });
  const forwardEnds: string[] = [];
  for (let mb = 0; mb < config.microbatches; mb++) {
    for (const layer of layers) {
      const rank = owner(layer.id);
      const boundary = layer.id > 0 && owner(layer.id - 1) !== rank;
      const input =
        layer.id > 0 ? id(boundary ? "recv-f" : "y", layer.id - 1, mb) : null;
      const a = buffer(
        id("a", layer.id, mb),
        layer.id,
        rank,
        payload(layer.inputWidth),
        "activations",
        `Saved input · microbatch ${mb + 1}`,
      );
      const y = buffer(
        id("y", layer.id, mb),
        layer.id,
        rank,
        payload(layer.outputWidth),
        "temporary",
        `Forward output · microbatch ${mb + 1}`,
      );
      const deps =
        layer.id === 0
          ? []
          : [
              end(
                boundary
                  ? `pp:send:f:${layer.id - 1}:${mb}`
                  : id("f", layer.id - 1, mb),
              ),
            ];
      const f = operation(
        "forward",
        id("f", layer.id, mb),
        [layer.id],
        [rank],
        deps,
        [bufferId("p", layer.id, rank), ...(input ? [input] : [])],
        [a, y],
        input ? [input] : [],
        {
          samples: batch,
          microbatch: mb,
          label: `forward L${layer.id + 1} · Stage ${rank} · MB ${mb + 1}`,
        },
      );
      forwardEnds.push(f.id);
      if (layer.id < layers.length - 1 && owner(layer.id + 1) !== rank) {
        const received = buffer(
          id("recv-f", layer.id, mb),
          layer.id,
          owner(layer.id + 1),
          payload(layer.outputWidth),
          "temporary",
          `Received activation · microbatch ${mb + 1}`,
        );
        operation(
          "send-recv",
          `pp:send:f:${layer.id}:${mb}`,
          [layer.id],
          [rank, owner(layer.id + 1)],
          [end(f.id)],
          [y],
          [received],
          [y],
          {
            logicalBytes: payload(layer.outputWidth),
            microbatch: mb,
            phaseTag: "pp-forward-transfer",
            label: `Activation transfer · Stage ${rank} → ${owner(layer.id + 1)} · MB ${mb + 1}`,
          },
        );
      }
    }
  }
  const backwardEnds: string[] = [];
  for (let mb = config.microbatches - 1; mb >= 0; mb--) {
    for (const layer of [...layers].reverse()) {
      const rank = owner(layer.id);
      const last = layer.id === layers.length - 1;
      const boundary = !last && owner(layer.id + 1) !== rank;
      const upstream = last
        ? id("y", layer.id, mb)
        : id(boundary ? "recv-b" : "dx", layer.id + 1, mb);
      const deps = last
        ? forwardEnds.map(end)
        : [
            end(
              boundary
                ? `pp:send:b:${layer.id + 1}:${mb}`
                : id("b", layer.id + 1, mb),
            ),
          ];
      const first = mb === config.microbatches - 1;
      const g = bufferId("g", layer.id, rank);
      if (first)
        buffer(
          g,
          layer.id,
          rank,
          layer.elements * 4,
          "gradients",
          "Accumulated global-batch layer gradient",
        );
      const dx = buffer(
        id("dx", layer.id, mb),
        layer.id,
        rank,
        payload(layer.inputWidth),
        "temporary",
        `Input gradient · microbatch ${mb + 1}`,
      );
      const a = id("a", layer.id, mb);
      const b = operation(
        "backward",
        id("b", layer.id, mb),
        [layer.id],
        [rank],
        deps,
        [bufferId("p", layer.id, rank), a, upstream, ...(first ? [] : [g])],
        [...(first ? [g] : []), dx],
        [a, upstream, ...(layer.id === 0 ? [dx] : [])],
        {
          samples: batch,
          microbatch: mb,
          label: `backward L${layer.id + 1} · Stage ${rank} · MB ${mb + 1}`,
        },
      );
      backwardEnds.push(b.id);
      if (layer.id > 0 && owner(layer.id - 1) !== rank) {
        const received = buffer(
          id("recv-b", layer.id, mb),
          layer.id,
          owner(layer.id - 1),
          payload(layer.inputWidth),
          "temporary",
          `Received input gradient · microbatch ${mb + 1}`,
        );
        operation(
          "send-recv",
          `pp:send:b:${layer.id}:${mb}`,
          [layer.id],
          [rank, owner(layer.id - 1)],
          [end(b.id)],
          [dx],
          [received],
          [dx],
          {
            logicalBytes: payload(layer.inputWidth),
            microbatch: mb,
            phaseTag: "pp-backward-transfer",
            label: `Gradient transfer · Stage ${rank} → ${owner(layer.id - 1)} · MB ${mb + 1}`,
          },
        );
      }
    }
  }
  for (const rank of ranks) {
    const owned = layers.filter((layer) => owner(layer.id) === rank);
    const gradients = owned.map((layer) => bufferId("g", layer.id, rank));
    operation(
      "adam",
      `adam:${rank}`,
      owned.map((layer) => layer.id),
      [rank],
      backwardEnds.map(end),
      owned.flatMap((layer) => [
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
