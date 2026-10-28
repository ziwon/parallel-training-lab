import type { Config, Operation } from "./types";

const collectiveNames = {
  "all-reduce": "AllReduce",
  "all-gather": "AllGather",
  "reduce-scatter": "ReduceScatter",
  "all-to-all": "AllToAll",
} as const;

export interface CollectiveLink {
  url: string | null;
  note: string;
  sizeMiB: number;
  operation: string;
}

export function collectiveLink(
  operation: Operation,
  config: Config,
): CollectiveLink {
  const sizeMiB = operation.logicalBytes / 1_048_576;
  if (!(operation.kind in collectiveNames)) {
    return {
      url: null,
      note:
        operation.kind === "send-recv"
          ? "This is a matched pipeline send/receive between two stages. NCCL Lab has no point-to-point operation in its URL contract."
          : "This operation is local computation.",
      sizeMiB,
      operation: operation.kind,
    };
  }
  const name = collectiveNames[operation.kind as keyof typeof collectiveNames];
  if (config.ranks === 1) {
    return {
      url: null,
      note: "One rank makes this collective a no-op. NCCL Lab requires at least two ranks.",
      sizeMiB,
      operation: name,
    };
  }
  if (
    !Number.isFinite(sizeMiB) ||
    sizeMiB < 4 / 1_048_576 ||
    sizeMiB > 1_048_576
  ) {
    return {
      url: null,
      note: "This logical buffer size is outside NCCL Lab's supported range.",
      sizeMiB,
      operation: name,
    };
  }
  const url = new URL("https://nccl-lab.pages.dev/");
  url.searchParams.set("op", name);
  url.searchParams.set("alg", name === "AllToAll" ? "Naive" : "Ring");
  url.searchParams.set("n", String(config.ranks));
  url.searchParams.set("size", String(sizeMiB));
  url.searchParams.set("root", "0");
  url.searchParams.set(
    "red",
    operation.reduction === "sum" || name === "AllGather" || name === "AllToAll"
      ? "Sum"
      : "Average",
  );
  const supportedBandwidth = [16, 32, 50, 100].includes(config.bandwidthGBps);
  if (supportedBandwidth)
    url.searchParams.set("bw", String(config.bandwidthGBps));
  const sizeMeaning =
    name === "AllReduce"
      ? "per-rank buffer"
      : name === "AllToAll"
        ? "representative per-rank input buffer (maximum modeled rank input)"
        : name === "AllGather"
          ? "total gathered output"
          : "total input before scattering";
  const bandwidthNote = supportedBandwidth
    ? "The bandwidth preset is preserved."
    : "This bandwidth is not an NCCL Lab preset, so its default 50 GB/s is used.";
  const routingNote =
    name === "AllToAll"
      ? " NCCL Lab distributes this illustrative buffer into equal peer chunks. This lab routes tokens with potentially unequal peer counts, so the external example does not reproduce the exact routing matrix or transferred bytes."
      : "";
  return {
    url: url.toString(),
    operation: name,
    sizeMiB,
    note: `${sizeMiB} MiB is the ${sizeMeaning}, converted from ${operation.logicalBytes} bytes. Both tools model FP32 storage. NCCL Lab uses illustrative scalar samples, not these training values.${routingNote} ${bandwidthNote} Its latency model is independent.`,
  };
}
