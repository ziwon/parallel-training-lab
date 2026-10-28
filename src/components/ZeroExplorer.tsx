import { ArrowLeftRight } from "lucide-react";
import type { Config, PrecisionProfile, ZeroStrategy } from "../engine/types";
import { zeroStateSummary, zeroStrategies } from "../engine/state-layout";
import { bytes } from "../format";
import { text as t } from "../text";

export function ZeroExplorer({
  config,
  pair,
  onPairChange,
  onPrecisionChange,
}: {
  config: Config;
  pair: [ZeroStrategy, ZeroStrategy];
  onPairChange: (pair: [ZeroStrategy, ZeroStrategy]) => void;
  onPrecisionChange: (precision: PrecisionProfile) => void;
}) {
  const summary = zeroStateSummary(config);
  return (
    <section className="zero-explorer" aria-label={t.zeroTitle}>
      <p className="zero-intro">{t.zeroIntro}</p>
      <div className="zero-controls">
        {[t.runA, t.runB].map((label, index) => (
          <label className="field" key={label}>
            <span>{t.zeroStageControl(label)}</span>
            <select
              value={pair[index]}
              onChange={(event) => {
                const next: [ZeroStrategy, ZeroStrategy] = [...pair];
                next[index] = event.target.value as ZeroStrategy;
                onPairChange(next);
              }}
            >
              {zeroStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {t.strategies[strategy]}
                </option>
              ))}
            </select>
          </label>
        ))}
        <label className="field zero-profile">
          <span>{t.zeroProfile}</span>
          <select
            value={config.precision}
            onChange={(event) =>
              onPrecisionChange(event.target.value as PrecisionProfile)
            }
          >
            <option value="fp32">{t.precisionProfiles.fp32}</option>
            <option value="bf16-master">
              {t.precisionProfiles["bf16-master"]}
            </option>
          </select>
        </label>
      </div>
      <p className="zero-profile-breakdown">
        {t.zeroProfileBreakdown[config.precision]}
      </p>
      <p className="zero-profile-note">{t.zeroProfileNote}</p>
      <div className="zero-summary-heading">
        <h2>{t.zeroSummary}</h2>
        <span>{t.rankOption(config.ranks)}</span>
      </div>
      <div
        className="zero-table-scroll"
        tabIndex={0}
        role="region"
        aria-label={t.zeroSummary}
      >
        <table className="zero-summary">
          <thead>
            <tr>
              <th scope="col">{t.strategy}</th>
              <th scope="col">{t.categories.parameters}</th>
              <th scope="col">{t.categories.gradients}</th>
              <th scope="col">{t.zeroMaster}</th>
              <th scope="col">{t.zeroMoments}</th>
              <th scope="col">{t.zeroTotal}</th>
              <th scope="col">{t.zeroPerParameter}</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => (
              <tr
                key={row.strategy}
                className={
                  pair.includes(row.strategy) ? "zero-selected-row" : ""
                }
              >
                <th scope="row">
                  {t.strategies[row.strategy]}
                  <span className="zero-run-markers">
                    {pair.map(
                      (strategy, index) =>
                        strategy === row.strategy && (
                          <span key={index}>{index === 0 ? "A" : "B"}</span>
                        ),
                    )}
                  </span>
                </th>
                <td>{bytes(row.parameterBytes)}</td>
                <td>{bytes(row.gradientBytes)}</td>
                <td>{bytes(row.masterBytes)}</td>
                <td>{bytes(row.optimizerBytes)}</td>
                <td>{bytes(row.totalBytes)}</td>
                <td>
                  {row.bytesPerParameter.toLocaleString("en-US", {
                    maximumFractionDigits: 2,
                  })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="zero-summary-note">{t.zeroSummaryNote}</p>
      <p className="zero-identity">
        <ArrowLeftRight size={15} />
        <span>{t.zeroIdentity}</span>
      </p>
    </section>
  );
}
