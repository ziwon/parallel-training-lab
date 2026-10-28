import { ArrowRight, SlidersHorizontal } from "lucide-react";
import type { Config } from "../engine/types";
import {
  expertForSample,
  expertOwner,
  stageOwner,
  workloadName,
} from "../engine/workload";
import { text as t } from "../text";

export function StrategyOverview({
  config,
  onConfigure,
}: {
  config: Config;
  onConfigure: () => void;
}) {
  const { strategy, ranks } = config;
  return (
    <section
      className={`strategy-overview strategy-${strategy}`}
      aria-label={t.layoutLabel}
    >
      <div className="strategy-explanation">
        <h2>{t.strategyShort[strategy]}</h2>
        <p>{t.strategyExplain[strategy]}</p>
        <span className="workload-tag">{workloadName(strategy)}</span>
        {strategy === "pp" && (
          <button className="text-button" onClick={onConfigure}>
            <SlidersHorizontal size={13} />
            {config.microbatches} {t.microbatches.toLowerCase()}
          </button>
        )}
      </div>
      <div
        className="partition-map"
        aria-label={t.ownershipDiagram(t.strategies[strategy])}
      >
        {Array.from({ length: ranks }, (_, rank) => (
          <div key={rank} className="partition-rank">
            <strong>{strategy === "pp" ? t.stage(rank) : t.rank(rank)}</strong>
            <div className={`partition-cells ${strategy}`}>
              {Array.from({ length: strategy === "ep" ? 8 : 4 }, (_, layer) => {
                const owned =
                  strategy === "pp"
                    ? stageOwner(layer, ranks) === rank
                    : strategy === "ep"
                      ? expertOwner(layer, ranks) === rank
                      : true;
                return (
                  <span
                    key={layer}
                    className={owned ? "partition-owned" : "partition-absent"}
                  >
                    {strategy === "ep"
                      ? t.expertLabel(layer)
                      : t.layerLabel(layer)}
                    {owned && (
                      <small>
                        {strategy === "tp"
                          ? layer % 2 === 0
                            ? t.partitionLabels.cols
                            : t.partitionLabels.rows
                          : strategy === "fsdp"
                            ? `1/${ranks}`
                            : t.partitionLabels.full}
                      </small>
                    )}
                  </span>
                );
              })}
            </div>
            <small>
              {strategy === "ep"
                ? t.routedTokens(
                    Array.from(
                      { length: config.globalBatch },
                      (_, id) => id,
                    ).filter(
                      (id) =>
                        expertOwner(expertForSample(id, config.seed), ranks) ===
                        rank,
                    ).length,
                  )
                : strategy === "pp"
                  ? t.microbatchCount(config.microbatches)
                  : strategy === "tp"
                    ? t.sharedSamples(config.globalBatch)
                    : t.localSamples(config.globalBatch / ranks)}
            </small>
          </div>
        ))}
        <div className="partition-caption">
          <ArrowRight size={13} />
          {t.partitionCaption[strategy]}
        </div>
      </div>
    </section>
  );
}
