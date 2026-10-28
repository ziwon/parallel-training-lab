import type { Category, Config, Layer, Operation } from "../types";
import { TOKENS_PER_SAMPLE } from "../model";
import { expertForSample, expertOwner } from "../workload";
import { bufferId, createPlan, end } from "./plan";

export function epPlan(config: Config, layers: Layer[]) {
  const { plan, ranks, allocate, operation } = createPlan(config, layers);
  const localBatch = config.globalBatch / config.ranks;
  const expertCounts = layers.map(() => 0);
  const routes = ranks.map(() => ranks.map(() => 0));
  for (let sample = 0; sample < config.globalBatch; sample++) {
    const expert = expertForSample(sample, config.seed);
    expertCounts[expert]++;
    routes[Math.floor(sample / localBatch)][
      expertOwner(expert, config.ranks)
    ]++;
  }
  const owned = (rank: number) =>
    layers.filter((layer) => expertOwner(layer.id, config.ranks) === rank);
  const received = ranks.map((rank) =>
    owned(rank).reduce((sum, layer) => sum + expertCounts[layer.id], 0),
  );
  const tokenBytes =
    layers[0].inputWidth * TOKENS_PER_SAMPLE * config.modelScale * 4;
  const buffer = (
    id: string,
    layer: number,
    rank: number,
    bytes: number,
    category: Category,
    label: string,
    initial = false,
  ) =>
    allocate(
      {
        id,
        layer,
        rank,
        bytes,
        category,
        ownership: "local",
        label,
        range: [0, bytes / 4],
      },
      initial,
    );
  const rankBuffers = (
    prefix: string,
    counts: number[],
    label: string,
    initial = false,
  ) =>
    ranks.map((rank) =>
      buffer(
        `${prefix}:${rank}`,
        -1,
        rank,
        counts[rank] * tokenBytes,
        "temporary",
        label,
        initial,
      ),
    );
  const sourceCounts = ranks.map(() => localBatch);
  const sourceInputs = rankBuffers(
    "ep:source-input",
    sourceCounts,
    "Source token input payload",
    true,
  );
  const remote =
    Math.max(
      0,
      ...ranks.flatMap((rank) => [
        routes[rank].reduce(
          (sum, count, peer) => sum + (peer === rank ? 0 : count),
          0,
        ),
        routes.reduce(
          (sum, row, peer) => sum + (peer === rank ? 0 : row[rank]),
          0,
        ),
      ]),
    ) * tokenBytes;
  function exchange(
    id: string,
    phaseTag: Operation["phaseTag"],
    prereqs: string[],
    reads: string[],
    outputs: string[],
    releases: string[],
    reverse = false,
  ) {
    return operation(
      "all-to-all",
      id,
      layers.map((layer) => layer.id),
      ranks,
      prereqs.map(end),
      reads,
      outputs,
      releases,
      {
        logicalBytes:
          Math.max(...(reverse ? received : sourceCounts)) * tokenBytes,
        transferredBytesPerRank: remote,
        duration:
          config.ranks === 1
            ? 0
            : ((config.ranks - 1) * config.latencyUs) / 1000 +
              remote / (config.bandwidthGBps * 1e6),
        phaseTag,
        label: `${phaseTag?.replaceAll("-", " ")} · variable token payloads`,
      },
    );
  }
  const dispatched = rankBuffers(
    "ep:received-input",
    received,
    "Dispatched expert input payload",
  );
  exchange(
    "ep:dispatch",
    "ep-dispatch",
    [],
    sourceInputs,
    dispatched,
    sourceInputs,
  );
  const forwardIds: string[] = [];
  const expertOutputs: string[] = [];
  for (const layer of layers) {
    const rank = expertOwner(layer.id, config.ranks);
    const count = expertCounts[layer.id];
    const a = buffer(
      bufferId("a", layer.id, rank),
      layer.id,
      rank,
      count * tokenBytes,
      "activations",
      `Saved expert input · ${count} tokens`,
    );
    const y = buffer(
      bufferId("y", layer.id, rank),
      layer.id,
      rank,
      count * tokenBytes,
      "temporary",
      `Expert output · ${count} tokens`,
    );
    const last = owned(rank).at(-1)!.id === layer.id;
    const f = operation(
      "forward",
      bufferId("f", layer.id, rank),
      [layer.id],
      [rank],
      [end("ep:dispatch")],
      [bufferId("p", layer.id, rank), dispatched[rank]],
      [a, y],
      last ? [dispatched[rank]] : [],
      {
        samples: count,
        label: `forward Expert ${layer.id} · Rank ${rank} · ${count} tokens`,
      },
    );
    forwardIds.push(f.id);
    expertOutputs.push(y);
  }
  const combined = rankBuffers(
    "ep:source-output",
    sourceCounts,
    "Expert outputs in source token order",
  );
  const outputGradients = rankBuffers(
    "ep:source-gradient",
    sourceCounts,
    "Globally normalized output gradients",
  );
  exchange(
    "ep:combine",
    "ep-combine",
    forwardIds,
    expertOutputs,
    [...combined, ...outputGradients],
    expertOutputs,
    true,
  );
  const dispatchedGradients = rankBuffers(
    "ep:received-gradient",
    received,
    "Dispatched output-gradient payload",
  );
  exchange(
    "ep:backward-dispatch",
    "ep-backward-dispatch",
    ["ep:combine"],
    [...combined, ...outputGradients],
    dispatchedGradients,
    [...combined, ...outputGradients],
  );
  const backwardIds: string[] = [];
  const inputGradients: string[] = [];
  for (const layer of layers) {
    const rank = expertOwner(layer.id, config.ranks);
    const count = expertCounts[layer.id];
    const a = bufferId("a", layer.id, rank);
    const g = buffer(
      bufferId("g", layer.id, rank),
      layer.id,
      rank,
      layer.elements * 4,
      "gradients",
      "Owned expert gradient summed over routed tokens",
    );
    const dx = buffer(
      bufferId("dx", layer.id, rank),
      layer.id,
      rank,
      count * tokenBytes,
      "temporary",
      `Expert input gradient · ${count} tokens`,
    );
    const last = owned(rank).at(-1)!.id === layer.id;
    const b = operation(
      "backward",
      bufferId("b", layer.id, rank),
      [layer.id],
      [rank],
      [end("ep:backward-dispatch")],
      [bufferId("p", layer.id, rank), a, dispatchedGradients[rank]],
      [g, dx],
      [a, ...(last ? [dispatchedGradients[rank]] : [])],
      {
        samples: count,
        label: `backward Expert ${layer.id} · Rank ${rank} · ${count} tokens`,
      },
    );
    backwardIds.push(b.id);
    inputGradients.push(dx);
  }
  const returned = rankBuffers(
    "ep:returned-input-gradient",
    sourceCounts,
    "Input gradients restored to source token order",
  );
  exchange(
    "ep:backward-combine",
    "ep-backward-combine",
    backwardIds,
    inputGradients,
    returned,
    [...inputGradients, ...returned],
    true,
  );
  for (const rank of ranks) {
    const local = owned(rank);
    operation(
      "adam",
      `adam:${rank}`,
      local.map((layer) => layer.id),
      [rank],
      [end("ep:backward-combine")],
      local.flatMap((layer) => [
        bufferId("p", layer.id, rank),
        bufferId("m", layer.id, rank),
        bufferId("g", layer.id, rank),
      ]),
      [],
      local.map((layer) => bufferId("g", layer.id, rank)),
    );
  }
  return plan;
}
