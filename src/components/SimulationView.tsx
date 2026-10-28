import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  Check,
  ChevronRight,
  Cpu,
  ExternalLink,
  Layers3,
  Network,
  ScanLine,
} from "lucide-react";
import {
  categories,
  type Category,
  type Operation,
  type Simulation,
  type Snapshot,
} from "../engine/types";
import { getNumericalSnapshot, type NumericalResult } from "../engine/numerics";
import { collectiveLink } from "../engine/nccl";
import {
  parameterIndices,
  expertForSample,
  expertOwner,
  batchDescription,
} from "../engine/workload";
import { bytes, count, ms } from "../format";
import { text as t } from "../text";

export interface Focus {
  rank: number;
  layer: number | null;
  category: Category | null;
  operation: string | null;
}
interface Props {
  run: Simulation;
  numerical: NumericalResult;
  snapshot: Snapshot;
  time: number;
  maxTime: number;
  focus: Focus;
  onFocus: (focus: Focus) => void;
  onSeek: (time: number) => void;
  onBoundary: (index: number) => void;
  label: string;
  compact?: boolean;
}
const kindAbbreviation = t.operationAbbreviations;
const groupLabel = (run: Simulation, layer: number) =>
  run.config.strategy === "ep" ? t.expertLabel(layer) : t.layerLabel(layer);

function stateOf(run: Simulation, snapshot: Snapshot, op: Operation) {
  if (snapshot.completed.includes(op.id)) return "complete";
  if (snapshot.executing.includes(op.id)) return "executing";
  return op.prerequisites.every(
    (dep) =>
      snapshot.completed.includes(dep.operation) ||
      (dep.phase === "start" && snapshot.executing.includes(dep.operation)),
  )
    ? "ready"
    : "waiting";
}

export function SimulationView({
  run,
  numerical,
  snapshot,
  time,
  maxTime,
  focus,
  onFocus,
  onSeek,
  onBoundary,
  label,
  compact,
}: Props) {
  const numericalState = getNumericalSnapshot(numerical, snapshot.eventIndex);
  const rank = Math.min(focus.rank, run.config.ranks - 1);
  const selected = run.operations.find((op) => op.id === focus.operation);
  const focusedLayers =
    selected?.layers ?? (focus.layer === null ? [] : [focus.layer]);
  const selectOperation = (op: Operation) =>
    onFocus({
      ...focus,
      operation: op.id,
      layer: op.layers[0] ?? null,
      rank: op.ranks.length === 1 ? op.ranks[0] : rank,
    });
  const resident = new Set(snapshot.resident);
  const lastSnapshot = run.snapshots[run.snapshots.length - 1];
  const allDone = snapshot.completed.length === run.operations.length;
  const executing = run.operations.filter((op) =>
    snapshot.executing.includes(op.id),
  );
  const currentPhase = allDone
    ? t.complete
    : snapshot.eventIndex === 0
      ? t.initialized
      : executing.some((op) => op.kind === "adam")
        ? t.adam
        : executing.some((op) => op.kind === "backward")
          ? t.backward
          : executing.some((op) => op.phaseTag === "tp-forward-sum")
            ? t.tpForwardPhase
            : executing.some((op) => op.phaseTag === "tp-backward-sum")
              ? t.tpBackwardPhase
              : executing.some((op) => op.phaseTag?.startsWith("ep-backward"))
                ? t.backward
                : executing.some(
                      (op) =>
                        op.kind === "all-reduce" ||
                        op.kind === "reduce-scatter",
                    )
                  ? t.sync
                  : executing.some((op) => op.id.startsWith("ag:b:"))
                    ? t.backward
                    : t.forward;
  return (
    <article
      className={`simulation ${compact ? "compact" : ""}`}
      aria-label={label}
    >
      <div className="run-heading">
        <span className="run-letter">{label}</span>
        <h2>
          {t.strategies[run.config.strategy]}{" "}
          <span>{t.strategyNames[run.config.strategy]}</span>
        </h2>
        <span className={`phase-tag ${allDone ? "done" : ""}`}>
          <span className="status-dot" />
          {currentPhase}
        </span>
      </div>
      <p className="run-batch-description">{batchDescription(run.config)}</p>
      <div className="metrics">
        <div>
          <span>{t.duration}</span>
          <strong>
            {ms(run.duration)} <small>{t.unitMs}</small>
          </strong>
          <small>{t.modeledTime}</small>
        </div>
        <div>
          <span>{t.peak}</span>
          <strong>
            {bytes(Math.max(...lastSnapshot.memory.map((m) => m.peakTotal)))}
          </strong>
          <small>
            {lastSnapshot.memory.some((m) => m.oom) ? t.oom : t.precision}
          </small>
        </div>
        <div>
          <span>{t.collectives}</span>
          <strong>
            {run.operations
              .filter((op) => op.lane === "collective")
              .length.toString()
              .padStart(2, "0")}
          </strong>
          <small>
            {run.config.ranks === 1
              ? t.noCollective
              : `${run.config.ranks} ${t.ranks.toLowerCase()}`}
          </small>
        </div>
        <div>
          <span>{t.parameters}</span>
          <strong>
            {count(run.layers.reduce((sum, l) => sum + l.elements, 0))}
          </strong>
          <small>{t.logicalModel}</small>
        </div>
      </div>
      <div className="workspace-panel">
        <section className="layer-section">
          <div className="section-heading">
            <h3>
              <Layers3 size={15} />
              {t.layerFlow}
            </h3>
            <span>{t.strategyShort[run.config.strategy]}</span>
          </div>
          <div
            className={`layer-flow ${run.config.strategy === "ep" ? "expert-flow" : ""}`}
          >
            <span className="flow-end">x</span>
            {run.layers.map((layer, index) => {
              const layerOps = run.operations.filter(
                (op) =>
                  op.layers.includes(layer.id) &&
                  (run.config.strategy === "pp" ||
                    run.config.strategy === "ep" ||
                    op.ranks[0] === rank),
              );
              const forwards = layerOps.filter((op) => op.kind === "forward");
              const backwards = layerOps.filter((op) => op.kind === "backward");
              const status =
                backwards.length &&
                backwards.every((op) => snapshot.completed.includes(op.id))
                  ? "backward-done"
                  : backwards.some((op) => snapshot.executing.includes(op.id))
                    ? "backward-active"
                    : forwards.length &&
                        forwards.every((op) =>
                          snapshot.completed.includes(op.id),
                        )
                      ? "forward-done"
                      : forwards.some((op) =>
                            snapshot.executing.includes(op.id),
                          )
                        ? "forward-active"
                        : "pending";
              return (
                <div className="layer-unit" key={layer.id}>
                  <button
                    className={`layer-node ${status} ${focusedLayers.includes(layer.id) ? "selected" : ""}`}
                    onClick={() =>
                      onFocus({ ...focus, layer: layer.id, operation: null })
                    }
                    aria-pressed={focusedLayers.includes(layer.id)}
                    title={`${layer.inputWidth} × ${layer.outputWidth} · ${count(layer.elements)} ${t.parameters.toLowerCase()}`}
                  >
                    <span>
                      {run.config.strategy === "ep"
                        ? `E${index}`
                        : t.layerLabel(index)}
                    </span>
                    <strong>{layer.name}</strong>
                    <small>{t.parameterCount(count(layer.elements))}</small>
                    <div className="layer-progress">
                      <i />
                      <i />
                    </div>
                  </button>
                  {run.config.strategy !== "ep" && (
                    <ArrowRight className="layer-arrow" size={14} />
                  )}
                </div>
              );
            })}
            <span className="flow-end">ℒ</span>
          </div>
        </section>
        <div className="rank-inspector-grid">
          <section className="ranks-section">
            <div className="section-heading">
              <h3>
                <Cpu size={15} />
                {t.rankOwnership}
              </h3>
              <button
                className="text-button"
                onClick={() =>
                  onFocus({
                    rank: 0,
                    layer: null,
                    category: null,
                    operation: null,
                  })
                }
              >
                {t.clearFocus}
              </button>
            </div>
            <div className={`rank-grid ranks-${run.config.ranks}`}>
              {snapshot.memory.map((memory) => {
                const numericalRank = numericalState.rankStates[memory.rank];
                const running = run.operations.filter(
                  (op) =>
                    snapshot.executing.includes(op.id) &&
                    op.ranks.includes(memory.rank),
                );
                const next = run.operations.find(
                  (op) =>
                    op.ranks.includes(memory.rank) &&
                    !snapshot.completed.includes(op.id) &&
                    !snapshot.executing.includes(op.id),
                );
                const status = running.length
                  ? running
                      .map(
                        (op) =>
                          `${kindAbbreviation[op.kind]} ${op.layers.map((l) => groupLabel(run, l)).join(", ")}`,
                      )
                      .join(" · ")
                  : allDone
                    ? t.complete
                    : next
                      ? `${t.statuses[stateOf(run, snapshot, next)]} · ${kindAbbreviation[next.kind]}`
                      : t.initialized;
                return (
                  <div
                    key={memory.rank}
                    className={`rank-card ${rank === memory.rank ? "selected" : ""} ${selected && !selected.ranks.includes(memory.rank) ? "dimmed" : ""}`}
                  >
                    <button
                      className="rank-card-heading"
                      aria-pressed={rank === memory.rank}
                      onClick={() => onFocus({ ...focus, rank: memory.rank })}
                    >
                      <Cpu size={16} />
                      <strong>{t.rank(memory.rank)}</strong>
                      <span
                        className={`rank-state ${running.length ? "running" : ""}`}
                      >
                        {status}
                      </span>
                    </button>
                    <div className="rank-meta">
                      <span>
                        {run.config.strategy === "ddp"
                          ? t.replica
                          : run.config.strategy === "fsdp"
                            ? t.shard(memory.rank, run.config.ranks)
                            : run.config.strategy === "tp"
                              ? t.tensorOwnership
                              : run.config.strategy === "pp"
                                ? `${t.stageOwnership} · ${memory.rank}`
                                : t.expertOwnership}
                      </span>
                      <span>
                        {run.config.strategy === "tp"
                          ? t.sameData
                          : run.config.strategy === "ep"
                            ? `${Array.from({ length: run.config.globalBatch }, (_, i) => i).filter((i) => expertOwner(expertForSample(i, run.config.seed), run.config.ranks) === memory.rank).length} routed tokens`
                            : t.sampleRange(numericalRank.sampleIds)}
                      </span>
                    </div>
                    <div className="tensor-grid">
                      {run.layers.map((layer) => {
                        const storage = run.allocations.find(
                          (a) =>
                            a.rank === memory.rank &&
                            a.layer === layer.id &&
                            a.category === "parameters",
                        );
                        if (!storage) return null;
                        const gathered =
                          run.config.strategy === "fsdp" &&
                          run.allocations.some(
                            (a) =>
                              a.rank === memory.rank &&
                              a.layer === layer.id &&
                              a.category === "temporary" &&
                              resident.has(a.id),
                          );
                        const hasGradient = run.allocations.some(
                          (a) =>
                            a.rank === memory.rank &&
                            a.layer === layer.id &&
                            a.category === "gradients" &&
                            resident.has(a.id),
                        );
                        return (
                          <button
                            key={layer.id}
                            className={`tensor-block ${focusedLayers.includes(layer.id) && (focus.category === null || focus.category === "parameters") ? "tensor-selected" : ""}`}
                            onClick={() =>
                              onFocus({
                                rank: memory.rank,
                                layer: layer.id,
                                category: "parameters",
                                operation: null,
                              })
                            }
                            title={`${groupLabel(run, layer.id)} ${storage?.label ?? t.parameters} · ${bytes(storage?.bytes ?? 0)}`}
                          >
                            <div>
                              <strong>
                                {run.config.strategy === "ep"
                                  ? `E${layer.id}`
                                  : groupLabel(run, layer.id)}
                              </strong>
                              <span>
                                {run.config.strategy === "ddp"
                                  ? t.categoryCodes.parameters
                                  : `${t.categoryCodes.parameters}${memory.rank}`}
                              </span>
                            </div>
                            <div className="shard-stripes">
                              {Array.from(
                                { length: run.config.ranks },
                                (_, r) => (
                                  <i
                                    key={r}
                                    className={
                                      ["ddp", "pp", "ep"].includes(
                                        run.config.strategy,
                                      ) || r === memory.rank
                                        ? "owned"
                                        : ""
                                    }
                                  />
                                ),
                              )}
                            </div>
                            <small>{bytes(storage?.bytes ?? 0)}</small>
                            <div className="buffer-flags">
                              <span
                                className={gathered ? "buffer-gathered" : ""}
                              >
                                {gathered
                                  ? t.bufferLabels.full
                                  : run.config.strategy === "ddp"
                                    ? t.bufferLabels.replica
                                    : run.config.strategy === "tp"
                                      ? layer.id % 2 === 0
                                        ? t.partitionLabels.columns
                                        : t.partitionLabels.rows
                                      : run.config.strategy === "pp"
                                        ? t.partitionLabels.stage
                                        : run.config.strategy === "ep"
                                          ? t.partitionLabels.expert
                                          : t.bufferLabels.shard}
                              </span>
                              {hasGradient && (
                                <span className="buffer-gradient">
                                  {t.categoryCodes.gradients}
                                </span>
                              )}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                    <div
                      className="memory-bar"
                      aria-label={`${t.memory} ${bytes(memory.total)}`}
                    >
                      {categories.map((category) => (
                        <button
                          key={category}
                          className={`memory-segment cat-${category} ${focus.category === category ? "segment-selected" : ""}`}
                          style={{
                            flexGrow: memory.current[category],
                            display: memory.current[category]
                              ? undefined
                              : "none",
                          }}
                          aria-label={`${t.categories[category]} ${bytes(memory.current[category])}`}
                          title={`${t.categories[category]} · ${bytes(memory.current[category])}`}
                          onClick={() =>
                            onFocus({
                              ...focus,
                              rank: memory.rank,
                              category,
                              operation: null,
                            })
                          }
                        />
                      ))}
                    </div>
                    <div className="rank-memory">
                      <span>
                        <strong>{bytes(memory.total)}</strong> {t.memoryUsed}
                      </span>
                      <span className={memory.oom ? "error-text" : ""}>
                        {memory.oom
                          ? t.oom
                          : t.peakValue(bytes(memory.peakTotal))}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="tensor-legend">
              {categories
                .filter((c) => c !== "scratch")
                .map((category) => (
                  <button
                    key={category}
                    aria-pressed={focus.category === category}
                    onClick={() =>
                      onFocus({ ...focus, category, operation: null })
                    }
                  >
                    <i className={`cat-${category}`} />
                    {t.categories[category]}
                  </button>
                ))}
            </div>
            {snapshot.memory.some((m) => m.oom) && (
              <div className="oom-banner" role="status">
                {t.oomNote}
              </div>
            )}
          </section>
          <Inspector
            run={run}
            numerical={numerical}
            snapshot={snapshot}
            focus={{ ...focus, rank }}
            selected={selected}
            onSelect={selectOperation}
            onSeek={onSeek}
            onBoundary={onBoundary}
          />
        </div>
      </div>
      <Timeline
        run={run}
        snapshot={snapshot}
        focus={focus}
        time={time}
        maxTime={maxTime}
        onSelect={selectOperation}
      />
      <MemoryChart
        run={run}
        snapshot={snapshot}
        rank={rank}
        time={time}
        maxTime={maxTime}
        focus={focus}
        onFocus={onFocus}
        onBoundary={onBoundary}
      />
      <section className="result-panel">
        <div>
          <Check size={16} />
          <h3>{t.referenceCheck}</h3>
          <span className="muted">{t.futureResult}</span>
        </div>
        <strong
          className={
            numerical.maxParameterError <= numerical.tolerance &&
            numerical.maxGradientError <= numerical.tolerance
              ? "success-text"
              : "error-text"
          }
        >
          {numerical.maxParameterError <= numerical.tolerance &&
          numerical.maxGradientError <= numerical.tolerance
            ? t.matches
            : t.mismatch}
        </strong>
        <dl>
          <div>
            <dt>{t.lossBefore}</dt>
            <dd>{numerical.lossBefore.toFixed(6)}</dd>
          </div>
          <div>
            <dt>{t.lossAfter}</dt>
            <dd>{numerical.lossAfter.toFixed(6)}</dd>
          </div>
          <div>
            <dt>{t.gradientError}</dt>
            <dd>{numerical.maxGradientError.toExponential(1)}</dd>
          </div>
          <div>
            <dt>{t.parameterError}</dt>
            <dd>{numerical.maxParameterError.toExponential(1)}</dd>
          </div>
          <div>
            <dt>{t.tolerance}</dt>
            <dd>{numerical.tolerance.toExponential(0)}</dd>
          </div>
        </dl>
      </section>
    </article>
  );
}

function Inspector({
  run,
  numerical,
  snapshot,
  focus,
  selected,
  onSelect,
  onSeek,
  onBoundary,
}: {
  run: Simulation;
  numerical: NumericalResult;
  snapshot: Snapshot;
  focus: Focus;
  selected?: Operation;
  onSelect: (op: Operation) => void;
  onSeek: (time: number) => void;
  onBoundary: (index: number) => void;
}) {
  const state = selected ? stateOf(run, snapshot, selected) : null;
  const link =
    selected?.lane === "collective"
      ? collectiveLink(selected, run.config)
      : null;
  const layer = focus.layer ?? 0;
  const numericalRank = getNumericalSnapshot(numerical, snapshot.eventIndex)
    .rankStates[focus.rank];
  const channel = parameterIndices(
    run.config.strategy,
    layer,
    focus.rank,
    run.config.ranks,
  )[0];
  const initial = numerical.fixture.initialParameters[layer]?.[channel];
  const now = numericalRank.parameters[layer]?.[0];
  const gradient = numericalRank.gradients[layer];
  const gradientSample =
    gradient?.[
      gradient.length === 64 && run.config.strategy === "fsdp" ? channel : 0
    ];
  return (
    <aside className="inspector">
      <div className="section-heading">
        <h3>
          <ScanLine size={15} />
          {t.inspector}
        </h3>
        {state && (
          <span className={`state-chip ${state}`}>{t.statuses[state]}</span>
        )}
      </div>
      <label className="event-picker">
        <span>{t.event}</span>
        <select
          value={snapshot.eventIndex}
          onChange={(event) => onBoundary(Number(event.target.value))}
          aria-label={t.event}
        >
          {run.events.map((event) => (
            <option key={event.index} value={event.index}>
              {String(event.index).padStart(2, "0")} · {ms(event.time)}{" "}
              {t.unitMs} ·{" "}
              {event.operation
                ? `${t.eventPhases[event.phase]} ${run.operations.find((op) => op.id === event.operation)?.label}`
                : t.initialized}
            </option>
          ))}
        </select>
      </label>
      {selected ? (
        <div className="operation-details">
          <div className={`operation-icon kind-${selected.kind}`}>
            {selected.lane === "collective" ? (
              <Network size={20} />
            ) : (
              <Activity size={20} />
            )}
          </div>
          <h3>{t.operationKinds[selected.kind]}</h3>
          <div className="operation-subtitle">
            {selected.layers.map((l) => groupLabel(run, l)).join(" + ")} ·{" "}
            {selected.ranks.length === 1
              ? t.rank(selected.ranks[0])
              : `${selected.ranks.length} ${t.ranks.toLowerCase()}`}
          </div>
          <p>
            {selected.phaseTag?.startsWith("tp-")
              ? t.tpSumWhy
              : (selected.kind === "forward" || selected.kind === "backward") &&
                  run.config.strategy === "tp"
                ? t.tpWhy
                : (selected.kind === "forward" ||
                      selected.kind === "backward") &&
                    run.config.strategy === "pp"
                  ? t.ppWhy
                  : (selected.kind === "forward" ||
                        selected.kind === "backward") &&
                      run.config.strategy === "ep"
                    ? t.epWhy
                    : t.operationWhy[selected.kind]}
          </p>
          {selected.microbatch !== undefined && (
            <p className="operation-context">
              {t.microbatchProgress(
                selected.microbatch + 1,
                run.config.microbatches,
              )}
            </p>
          )}
          {selected.reduction && (
            <p className="operation-context">
              {t.reductionLabel(selected.reduction)}
            </p>
          )}
          <div className="operation-times">
            <div>
              <span>{t.start}</span>
              <strong>{ms(selected.start)}</strong>
            </div>
            <ArrowRight size={14} />
            <div>
              <span>{t.end}</span>
              <strong>
                {ms(selected.end)} <small>{t.unitMs}</small>
              </strong>
            </div>
          </div>
          <div className="jump-buttons">
            <button
              onClick={() => {
                const event = run.events.find(
                  (e) => e.operation === selected.id && e.phase === "start",
                );
                event ? onBoundary(event.index) : onSeek(selected.start);
              }}
            >
              {t.jumpStart}
            </button>
            <button
              onClick={() => {
                const event = run.events.find(
                  (e) => e.operation === selected.id && e.phase === "end",
                );
                event ? onBoundary(event.index) : onSeek(selected.end);
              }}
            >
              {t.jumpEnd}
            </button>
          </div>
          <details className="dependency-details" open>
            <summary>
              {t.prerequisites} <span>{selected.prerequisites.length}</span>
            </summary>
            <p className="small">
              {state === "waiting"
                ? t.waitingExplanation
                : state === "ready"
                  ? t.readyExplanation
                  : state === "executing"
                    ? t.executingExplanation
                    : t.completeExplanation}
            </p>
            <ul>
              {selected.prerequisites.map((dep, i) => {
                const other = run.operations.find(
                  (op) => op.id === dep.operation,
                )!;
                const ready =
                  snapshot.completed.includes(dep.operation) ||
                  (dep.phase === "start" &&
                    snapshot.executing.includes(dep.operation));
                return (
                  <li key={`${dep.operation}-${i}`}>
                    <button
                      onClick={() => onSelect(other)}
                      className={ready ? "satisfied" : ""}
                    >
                      {ready ? (
                        <Check size={12} />
                      ) : (
                        <span className="wait-dot" />
                      )}
                      <span>
                        {other.label} <small>{t.eventPhases[dep.phase]}</small>
                      </span>
                      <ChevronRight size={12} />
                    </button>
                  </li>
                );
              })}
            </ul>
            {selected.prerequisites.length === 0 && (
              <p className="small">{t.initialReady}</p>
            )}
          </details>
          {link && (
            <div className="collective-details">
              <dl>
                <div>
                  <dt>{t.queue}</dt>
                  <dd>{(selected.collectiveIndex ?? 0) + 1}</dd>
                </div>
                <div>
                  <dt>{t.logicalSize}</dt>
                  <dd>{bytes(selected.logicalBytes)}</dd>
                </div>
                <div>
                  <dt>{t.transferred}</dt>
                  <dd>{bytes(selected.transferredBytesPerRank)}</dd>
                </div>
              </dl>
              <details>
                <summary>{t.rankReadiness}</summary>
                {selected.ranks.map((r) => (
                  <div className="readiness-row" key={r}>
                    <span>{t.rank(r)}</span>
                    <span>
                      {t.readyAt}{" "}
                      {ms(selected.readyByRank[r] ?? selected.inputReadyAt)}{" "}
                      {t.unitMs}
                    </span>
                  </div>
                ))}
              </details>
              {link.url && (
                <a
                  className="nccl-link"
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {t.explore}
                  <ExternalLink size={13} />
                </a>
              )}
              <p className="small muted">{link.note}</p>
            </div>
          )}
          <details>
            <summary>{t.ledger}</summary>
            {(["allocate", "release"] as const).map((key) => (
              <div key={key}>
                <h4>{key === "allocate" ? t.allocated : t.released}</h4>
                {selected[key].map((id) => {
                  const allocation = run.allocations.find((a) => a.id === id)!;
                  return (
                    <div className="allocation-row" key={id}>
                      <i className={`cat-${allocation.category}`} />
                      <span>
                        {t.rank(allocation.rank)} · {allocation.label}
                      </span>
                      <b>{bytes(allocation.bytes)}</b>
                    </div>
                  );
                })}
              </div>
            ))}
            {selected.allocate.length + selected.release.length === 0 && (
              <p className="small">{t.noChanges}</p>
            )}
          </details>
        </div>
      ) : (
        <div className="empty-inspector">
          <Network size={26} />
          <h3>
            {focus.layer === null
              ? t.inspectDefault
              : `${groupLabel(run, layer)} · ${run.layers[layer].name}`}
          </h3>
          <p>{t.inspectDefaultNote}</p>
          <button
            className="suggested-event"
            onClick={() => {
              const op = run.operations.find(
                (op) =>
                  op.lane === "collective" &&
                  (focus.layer === null || op.layers.includes(layer)),
              );
              if (op) onSelect(op);
            }}
          >
            {t.collectives}
            <ArrowRight size={15} />
          </button>
        </div>
      )}
      <div className="numerical-inspector">
        <h4>{t.numerics}</h4>
        <span className="mono micro">
          {t.rank(focus.rank)} · {groupLabel(run, layer)} ·{" "}
          {channel === undefined ? t.notOwned : `w[${channel}]`}
        </span>
        <dl>
          <div>
            <dt>{t.initial}</dt>
            <dd>{initial?.toFixed(5) ?? "—"}</dd>
          </div>
          <div>
            <dt>{t.now}</dt>
            <dd>{now?.toFixed(5) ?? "—"}</dd>
          </div>
          <div>
            <dt>
              {numericalRank.synchronizedLayers.includes(layer)
                ? t.synchronized
                : t.localGradient}
            </dt>
            <dd>
              {gradientSample === undefined ? "·" : gradientSample.toFixed(5)}
            </dd>
          </div>
        </dl>
        <p className="small muted">{t.sampleNotice}</p>
      </div>
    </aside>
  );
}

function Timeline({
  run,
  snapshot,
  focus,
  time,
  maxTime,
  onSelect,
}: {
  run: Simulation;
  snapshot: Snapshot;
  focus: Focus;
  time: number;
  maxTime: number;
  onSelect: (op: Operation) => void;
}) {
  const end = Math.max(maxTime, 0.001);
  const focusedLayers =
    run.operations.find((op) => op.id === focus.operation)?.layers ??
    (focus.layer === null ? [] : [focus.layer]);
  const rows = [
    ...Array.from({ length: run.config.ranks }, (_, rank) => ({
      label: t.computeLane(rank),
      operations: run.operations.filter(
        (op) => op.lane === "compute" && op.ranks[0] === rank,
      ),
    })),
    {
      label: t.commLane,
      operations: run.operations.filter((op) => op.lane === "collective"),
    },
  ];
  return (
    <section className="timeline-panel panel">
      <div className="section-heading">
        <h3>
          <Activity size={15} />
          {t.timeline}
        </h3>
        <div className="timeline-key">
          <span className="key-forward">{t.legendForward}</span>
          <span className="key-backward">{t.legendBackward}</span>
          <span className="key-adam">{t.legendAdam}</span>
          <span className="key-collective">{t.legendComm}</span>
        </div>
      </div>
      <p className="panel-note">{t.timelineHint}</p>
      <div className="timeline-scroll" tabIndex={0} aria-label={t.timeline}>
        <div className="timeline-canvas">
          <div className="timeline-ruler">
            <span />
            {Array.from({ length: 6 }, (_, i) => (
              <span
                key={i}
                style={{ left: `calc(94px + (100% - 100px) * ${i / 5})` }}
              >
                {ms((end * i) / 5)}
                {i === 5 ? ` ${t.unitMs}` : ""}
              </span>
            ))}
          </div>
          {rows.map((row, index) => (
            <div
              className={`timeline-row ${index === run.config.ranks ? "comm-row" : ""} ${index === focus.rank ? "focused-lane" : ""}`}
              key={row.label}
            >
              <span className="lane-label">
                {index === run.config.ranks ? (
                  <Network size={12} />
                ) : (
                  <Cpu size={12} />
                )}{" "}
                {row.label}
              </span>
              <div className="lane-track">
                {Array.from({ length: 6 }, (_, i) => (
                  <i
                    className="grid-line"
                    key={i}
                    style={{ left: `${i * 20}%` }}
                  />
                ))}
                {row.operations.map((op) => (
                  <button
                    key={op.id}
                    className={`timeline-event kind-${op.kind} ${stateOf(run, snapshot, op)} ${focus.operation === op.id ? "selected" : ""} ${focusedLayers.length > 0 && !op.layers.some((layer) => focusedLayers.includes(layer)) ? "unrelated" : ""}`}
                    style={{
                      left: `${(op.start / end) * 100}%`,
                      width: `${Math.max(0.35, ((op.end - op.start) / end) * 100)}%`,
                    }}
                    title={t.operationTimeTitle(
                      op.label,
                      ms(op.start),
                      ms(op.end),
                    )}
                    aria-label={t.operationTimeLabel(
                      op.label,
                      ms(op.start),
                      ms(op.end),
                    )}
                    onClick={() => onSelect(op)}
                  >
                    {kindAbbreviation[op.kind]}
                    {op.layers.length === 1
                      ? ` ${run.config.strategy === "ep" ? `E${op.layers[0]}` : `L${op.layers[0] + 1}`}`
                      : ""}
                    {op.microbatch !== undefined
                      ? ` M${op.microbatch + 1}`
                      : ""}
                  </button>
                ))}
                <div
                  className="time-cursor"
                  style={{ left: `${Math.min(100, (time / end) * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function MemoryChart({
  run,
  snapshot,
  rank,
  time,
  maxTime,
  focus,
  onFocus,
  onBoundary,
}: {
  run: Simulation;
  snapshot: Snapshot;
  rank: number;
  time: number;
  maxTime: number;
  focus: Focus;
  onFocus: (focus: Focus) => void;
  onBoundary: (index: number) => void;
}) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(660);
  useEffect(() => {
    const node = plotRef.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const nextWidth = entries[0]?.contentRect.width;
      if (nextWidth) setWidth(Math.max(280, Math.round(nextWidth)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const height = 200,
    left = 72,
    top = 12,
    bottom = 26;
  const plotW = width - left - 12,
    plotH = height - top - bottom;
  const capacity = run.config.capacityMiB * 1048576;
  const maximum = Math.max(
    ...run.snapshots.map((s) => s.memory[rank].total),
    1,
  );
  const yMax = Math.max(
    maximum * 1.12,
    capacity <= maximum * 1.5 ? capacity * 1.05 : 0,
  );
  const end = Math.max(maxTime, 0.001);
  const x = (v: number) => left + (v / end) * plotW;
  const y = (v: number) => top + plotH - (v / yMax) * plotH;
  const points = [
    ...run.snapshots.map((s) => ({ ...s, time: s.time })),
    { ...run.snapshots.at(-1)!, time: end },
  ];
  const memory = snapshot.memory[rank];
  return (
    <section className="memory-panel panel">
      <div className="section-heading">
        <h3>
          <Layers3 size={15} />
          {t.memoryTimeline}
        </h3>
        <span className="selected-rank-label">
          {t.rank(rank)} · {bytes(memory.total)} {t.current.toLowerCase()}
        </span>
      </div>
      <p className="panel-note">{t.memoryHint}</p>
      <div className="memory-content">
        <div className="memory-plot" ref={plotRef}>
          <svg
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={t.memoryChart}
          >
            {Array.from({ length: 4 }, (_, i) => {
              const value = (yMax * i) / 3;
              return (
                <g key={i}>
                  <line
                    x1={left}
                    x2={width - 12}
                    y1={y(value)}
                    y2={y(value)}
                    className="chart-grid"
                  />
                  <text x={left - 8} y={y(value) + 3} textAnchor="end">
                    {bytes(value)}
                  </text>
                </g>
              );
            })}
            {categories.map((category, ci) => {
              const bottomValues = points.map((p) =>
                categories
                  .slice(0, ci)
                  .reduce((sum, c) => sum + p.memory[rank].current[c], 0),
              );
              const topValues = points.map(
                (p, i) => bottomValues[i] + p.memory[rank].current[category],
              );
              let d = `M ${x(points[0].time)} ${y(topValues[0])}`;
              for (let i = 1; i < points.length; i++)
                d += ` H ${x(points[i].time)} V ${y(topValues[i])}`;
              d += ` L ${x(points.at(-1)!.time)} ${y(bottomValues.at(-1)!)}`;
              for (let i = points.length - 2; i >= 0; i--)
                d += ` V ${y(bottomValues[i])} H ${x(points[i].time)}`;
              return (
                <path
                  key={category}
                  d={`${d} Z`}
                  className={`area-${category} ${focus.category && focus.category !== category ? "muted-area" : ""}`}
                />
              );
            })}
            {capacity <= yMax && (
              <g>
                <line
                  x1={left}
                  x2={width - 12}
                  y1={y(capacity)}
                  y2={y(capacity)}
                  className="capacity-line"
                />
                <text x={width - 14} y={y(capacity) - 5} textAnchor="end">
                  {t.capacityLabel} {bytes(capacity)}
                </text>
              </g>
            )}
            <line
              x1={x(time)}
              x2={x(time)}
              y1={top}
              y2={height - bottom}
              className="chart-cursor"
            />
            {(width < 480 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map(
              (fraction) => (
                <text
                  key={fraction}
                  x={x(end * fraction)}
                  y={height - 7}
                  textAnchor={fraction === 1 ? "end" : "middle"}
                >
                  {ms(end * fraction)} {t.unitMs}
                </text>
              ),
            )}
          </svg>
          <div className="memory-cause">
            <span>
              {t.event} {snapshot.eventIndex.toString().padStart(2, "0")}
            </span>
            <button onClick={() => onBoundary(snapshot.eventIndex)}>
              {run.events[snapshot.eventIndex].operation
                ? `${t.eventPhases[run.events[snapshot.eventIndex].phase]} · ${run.operations.find((op) => op.id === run.events[snapshot.eventIndex].operation)?.label}`
                : t.initialized}
              <ArrowDown size={12} />
            </button>
          </div>
        </div>
        <div className="memory-table">
          <table>
            <thead>
              <tr>
                <th>{t.category}</th>
                <th>{t.current}</th>
                <th>{t.peakCategory}</th>
              </tr>
            </thead>
            <tbody>
              {categories.map((category) => (
                <tr
                  key={category}
                  className={
                    focus.category === category ? "selected-category" : ""
                  }
                >
                  <td>
                    <button
                      onClick={() =>
                        onFocus({ ...focus, rank, category, operation: null })
                      }
                    >
                      <i className={`cat-${category}`} />
                      {t.categories[category]}
                    </button>
                  </td>
                  <td>{bytes(memory.current[category])}</td>
                  <td>{bytes(memory.peak[category])}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th>{t.totalPeak}</th>
                <td>{bytes(memory.total)}</td>
                <td>{bytes(memory.peakTotal)}</td>
              </tr>
            </tfoot>
          </table>
          <p className="small muted">{t.categoryPeakHint}</p>
        </div>
      </div>
    </section>
  );
}
