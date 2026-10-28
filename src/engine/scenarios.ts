import { parseConfig } from "./config";
import type { Config, ScenarioId, SessionConfig } from "./types";

export const scenarioIds = [
  "strategies",
  "reshard",
  "prefetch",
  "slow",
  "tensor",
  "pipeline",
  "experts",
] as const;

export function resolveSessionRuns(session: SessionConfig): Config[] {
  if (session.scenario === "zero") {
    return session.zeroPair.map((strategy) =>
      parseConfig({ ...session.config, strategy }),
    );
  }
  return session.compare
    ? scenarioPair(session.config, session.scenario).map(parseConfig)
    : [parseConfig(session.config)];
}

export function scenarioPair(
  base: Config,
  scenario: ScenarioId,
): [Config, Config] {
  const config = parseConfig(base);
  switch (scenario) {
    case "tensor":
      return [
        parseConfig({ ...config, strategy: "fsdp" }),
        parseConfig({ ...config, strategy: "tp" }),
      ];
    case "pipeline": {
      const pipeline = parseConfig({
        ...config,
        strategy: "pp",
        ranks: Math.min(config.ranks, 4),
        slowRank:
          config.slowRank !== null && config.slowRank >= 4
            ? null
            : config.slowRank,
      });
      return [{ ...pipeline, microbatches: 1 }, pipeline];
    }
    case "experts":
      return [
        { ...config, strategy: "ep", ranks: 1, slowRank: null },
        parseConfig({ ...config, strategy: "ep" }),
      ];
    case "strategies":
      return [
        { ...config, strategy: "ddp" },
        { ...config, strategy: "fsdp" },
      ];
    case "reshard":
      return [
        { ...config, strategy: "fsdp", reshardAfterForward: true, prefetch: 0 },
        {
          ...config,
          strategy: "fsdp",
          reshardAfterForward: false,
          prefetch: 0,
        },
      ];
    case "prefetch":
      return [
        { ...config, strategy: "fsdp", reshardAfterForward: true, prefetch: 0 },
        { ...config, strategy: "fsdp", reshardAfterForward: true, prefetch: 1 },
      ];
    case "slow":
      return [
        { ...config, slowRank: null },
        { ...config, slowRank: config.ranks - 1, slowMultiplier: 2 },
      ];
  }
}
