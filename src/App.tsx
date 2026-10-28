import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Activity,
  ArrowLeftRight,
  BookOpen,
  Check,
  ChevronRight,
  Cpu,
  Download,
  ExternalLink,
  Gauge,
  Layers3,
  Menu,
  Network,
  Pause,
  Play,
  RotateCcw,
  Share2,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { defaultConfig, parseConfig } from "./engine/config";
import { simulate, snapshotAtTime } from "./engine/simulation";
import { executeNumerics } from "./engine/numerics";
import {
  createExport,
  decodeSession,
  encodeSession,
  importExport,
} from "./engine/serialization";
import { scenarioIds, resolveSessionRuns } from "./engine/scenarios";
import type {
  Config,
  ScenarioId,
  SessionConfig,
  ZeroStrategy,
} from "./engine/types";
import { SimulationView, type Focus } from "./components/SimulationView";
import { strategies, batchDescription } from "./engine/workload";
import { StrategyOverview } from "./components/StrategyOverview";
import { ZeroExplorer } from "./components/ZeroExplorer";
import { ms } from "./format";
import { text as t } from "./text";

const initialFocus: Focus = {
  rank: 0,
  layer: null,
  category: null,
  operation: null,
};
const scenarioIcons = {
  strategies: ArrowLeftRight,
  reshard: Layers3,
  prefetch: Zap,
  slow: Gauge,
  tensor: Network,
  pipeline: Layers3,
  experts: Cpu,
};
const configLabels: Partial<Record<keyof Config, string>> = {
  strategy: t.strategy,
  ranks: t.ranks,
  microbatches: t.microbatches,
  reshardAfterForward: t.reshard,
  prefetch: t.prefetch,
  slowRank: t.slowRank,
  slowMultiplier: t.multiplier,
};

function loadSession(): { session: SessionConfig; error: string } {
  try {
    return { session: decodeSession(window.location.search), error: "" };
  } catch (error) {
    return {
      session: {
        config: defaultConfig,
        compare: false,
        scenario: "strategies",
      },
      error: `${t.invalidConfig} ${String(error instanceof Error ? error.message : error)}`,
    };
  }
}
function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
function Modal({
  title,
  children,
  onClose,
  wide,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`modal ${wide ? "wide" : ""}`}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      <div className="modal-heading">
        <h2>{title}</h2>
        <button className="icon-button" aria-label={t.close} onClick={onClose}>
          <X size={19} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

export default function App() {
  const [initial] = useState(loadSession);
  const [session, setSession] = useState(initial.session);
  const [notice, setNotice] = useState(initial.error);
  const [isError, setIsError] = useState(Boolean(initial.error));
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [focusedRun, setFocusedRun] = useState(0);
  const [boundaries, setBoundaries] = useState<(number | null)[]>([0, 0]);
  const [focuses, setFocuses] = useState<Focus[]>([
    { ...initialFocus },
    { ...initialFocus },
  ]);
  const [settings, setSettings] = useState(false);
  const [guide, setGuide] = useState(false);
  const [navigation, setNavigation] = useState(false);
  const [shareUrl, setShareUrl] = useState("");
  const upload = useRef<HTMLInputElement>(null);
  const isZero = session.scenario === "zero";
  const configs = useMemo(() => resolveSessionRuns(session), [session]);
  const runs = useMemo(() => configs.map(simulate), [configs]);
  const numerical = useMemo(() => runs.map(executeNumerics), [runs]);
  const maxTime = Math.max(...runs.map((run) => run.duration));
  const activeRun = Math.min(focusedRun, runs.length - 1);
  const snapshots = runs.map((run, i) =>
    boundaries[i] === null
      ? snapshotAtTime(run, time)
      : run.snapshots[Math.min(boundaries[i]!, run.snapshots.length - 1)],
  );
  const reset = useCallback(() => {
    setPlaying(false);
    setTime(0);
    setBoundaries([0, 0]);
  }, []);
  const updateSession = (next: SessionConfig) => {
    try {
      const validated = { ...next, config: parseConfig(next.config) };
      resolveSessionRuns(validated).forEach(parseConfig);
      setSession(validated);
      reset();
      setFocuses([{ ...initialFocus }, { ...initialFocus }]);
      setFocusedRun(0);
      setNotice("");
    } catch (error) {
      setNotice(
        `${t.invalidConfig} ${error instanceof Error ? error.message : String(error)}`,
      );
      setIsError(true);
    }
  };
  const seek = (value: number) => {
    setPlaying(false);
    setTime(Math.max(0, Math.min(maxTime, value)));
    setBoundaries([null, null]);
  };
  const boundary = (runIndex: number, eventIndex: number) => {
    const run = runs[runIndex];
    const event =
      run.events[Math.max(0, Math.min(run.events.length - 1, eventIndex))];
    setPlaying(false);
    setTime(event.time);
    setBoundaries(runIndex === 0 ? [event.index, null] : [null, event.index]);
    setFocusedRun(runIndex);
    if (event.operation) {
      const op = run.operations.find((item) => item.id === event.operation)!;
      setFocuses((previous) =>
        previous.map((focus, i) =>
          i === runIndex
            ? {
                ...focus,
                operation: op.id,
                layer: op.layers[0] ?? null,
                rank: op.ranks.length === 1 ? op.ranks[0] : focus.rank,
              }
            : focus,
        ),
      );
    }
  };
  const step = (direction: number) =>
    boundary(activeRun, snapshots[activeRun].eventIndex + direction);
  const togglePlay = () => {
    if (time >= maxTime) {
      setTime(0);
      setBoundaries([0, 0]);
    }
    setPlaying((value) => !value);
  };
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setBoundaries([null, null]);
      setTime((previous) =>
        Math.min(maxTime, previous + (maxTime / 240) * speed),
      );
    }, 50);
    return () => window.clearInterval(timer);
  }, [playing, maxTime, speed]);
  useEffect(() => {
    if (time >= maxTime) setPlaying(false);
  }, [time, maxTime]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (
        settings ||
        guide ||
        shareUrl ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        (event.target instanceof HTMLElement &&
          event.target.closest(
            'input,select,textarea,button,a,[contenteditable="true"],[role="region"][tabindex]',
          ))
      )
        return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlay();
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        step(1);
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        step(-1);
      }
      if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        reset();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const notify = (message: string, error = false) => {
    setNotice(message);
    setIsError(error);
  };
  const share = async () => {
    const url = new URL(window.location.href);
    url.search = encodeSession(session);
    url.hash = "";
    try {
      await navigator.clipboard.writeText(url.toString());
      notify(t.copied);
    } catch {
      setShareUrl(url.toString());
    }
  };
  const exportJSON = () => {
    const exported = {
      ...createExport(runs[0], session, numerical[0]),
      ...(session.compare
        ? { comparison: createExport(runs[1], session, numerical[1]) }
        : {}),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(exported, null, 2)], {
        type: "application/json",
      }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "parallel-viz-trace.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify(t.exported);
  };
  const importJSON = async (file?: File) => {
    if (!file) return;
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error(t.fileLimit);
      const next = importExport(await file.text());
      updateSession(next);
      notify(t.imported);
    } catch (error) {
      notify(
        `${t.invalidImport} ${error instanceof Error ? error.message : String(error)}`,
        true,
      );
    }
    if (upload.current) upload.current.value = "";
  };
  const standardSession = (
    compare: boolean,
    scenario: ScenarioId = session.scenario === "zero"
      ? "strategies"
      : session.scenario,
    strategy: Config["strategy"] = strategies.includes(
      session.config.strategy as (typeof strategies)[number],
    )
      ? session.config.strategy
      : "ddp",
  ): SessionConfig => ({
    config: { ...session.config, strategy, precision: "fp32" },
    compare,
    scenario,
  });
  const chooseScenario = (scenario: ScenarioId) =>
    updateSession(standardSession(true, scenario));
  const chooseZero = () => {
    updateSession({
      config: { ...session.config, strategy: "ddp" },
      compare: true,
      scenario: "zero",
      zeroPair:
        session.scenario === "zero" ? session.zeroPair : ["ddp", "zero3"],
    });
    setNavigation(false);
  };
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        {t.simulator}
      </a>
      {navigation && (
        <button
          className="nav-backdrop"
          aria-label={t.close}
          onClick={() => setNavigation(false)}
        />
      )}
      <aside className={`sidebar ${navigation ? "open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            reset();
          }}
        >
          <span className="brand-icon">
            <Network size={23} />
          </span>
          <strong>
            {t.brandName}
            <span>{t.brandSuffix}</span>
          </strong>
          <span className="brand-dot" />
        </a>
        <div className="workspace-label">
          <span>{t.brandMark}</span>
          <div>
            {t.workspace}
            <small>{t.title}</small>
          </div>
        </div>
        <nav aria-label={t.workspace}>
          <button
            className={!session.compare ? "active" : ""}
            onClick={() => {
              updateSession(standardSession(false));
              setNavigation(false);
            }}
          >
            <Activity size={17} />
            {t.simulator}
          </button>
          <button
            className={session.compare && !isZero ? "active" : ""}
            onClick={() => {
              updateSession(standardSession(true));
              setNavigation(false);
            }}
          >
            <ArrowLeftRight size={17} />
            {t.compare}
          </button>
          <button className={isZero ? "active" : ""} onClick={chooseZero}>
            <Layers3 size={17} />
            {t.zeroTitle}
          </button>
          <button
            onClick={() => {
              setGuide(true);
              setNavigation(false);
            }}
          >
            <BookOpen size={17} />
            {t.guide}
          </button>
        </nav>
        <div className="sidebar-section-label">{t.strategy}</div>
        <div className="sidebar-strategies">
          {strategies.map((strategy) => (
            <button
              key={strategy}
              className={
                session.config.strategy === strategy && !session.compare
                  ? "chosen"
                  : ""
              }
              onClick={() => {
                updateSession(standardSession(false, "strategies", strategy));
                setNavigation(false);
              }}
            >
              <Layers3 size={17} />
              <span>
                {t.strategies[strategy]}
                <small>{t.strategyShort[strategy]}</small>
              </span>
              {session.config.strategy === strategy && !session.compare && (
                <span className="status-dot" />
              )}
            </button>
          ))}
        </div>
        <div className="sidebar-note">
          <div className="mini-ranks">
            {[0, 1, 2, 3].map((r) => (
              <span key={r}>
                <Cpu size={13} />
              </span>
            ))}
          </div>
          <p>
            {isZero
              ? t.zeroIdentity
              : t.strategyExplain[session.config.strategy]}
          </p>
          <button onClick={() => setGuide(true)}>
            {t.guide}
            <ArrowLeftRight size={12} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <a
            href="https://nccl-lab.pages.dev/"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t.companion}
            <ExternalLink size={13} />
          </a>
          <div>
            <span className="status-dot" />
            {t.browserOnly}
            <small>{t.version}</small>
          </div>
          <p>{t.noGPU}</p>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <button
              className="icon-button mobile-menu"
              aria-label={t.workspace}
              onClick={() => setNavigation(true)}
            >
              <Menu size={19} />
            </button>
            <span>{t.workspace}</span>
            <ChevronRight size={13} />
            <strong>
              {isZero ? t.zeroTitle : session.compare ? t.compare : t.simulator}
            </strong>
          </div>
          <span className="educational">
            <span className="status-dot" />
            {t.educational}
          </span>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <h1>{isZero ? t.zeroTitle : t.title}</h1>
              <p>{isZero ? t.zeroSubtitle : t.subtitle}</p>
            </div>
            <div className="file-actions">
              <button
                title={t.import}
                aria-label={t.import}
                onClick={() => upload.current?.click()}
              >
                <Upload size={15} />
                <span>{t.import}</span>
              </button>
              <button aria-label={t.export} onClick={exportJSON}>
                <Download size={15} />
                <span>{t.export}</span>
              </button>
              <button aria-label={t.share} onClick={share}>
                <Share2 size={15} />
                <span>{t.share}</span>
              </button>
            </div>
            <input
              ref={upload}
              hidden
              tabIndex={-1}
              type="file"
              accept="application/json,.json"
              aria-label={t.import}
              onChange={(event) => void importJSON(event.target.files?.[0])}
            />
          </div>
          {notice && (
            <div
              role={isError ? "alert" : "status"}
              className={`notice ${isError ? "error" : ""}`}
            >
              <span>{notice}</span>
              <button
                className="icon-button"
                onClick={() => setNotice("")}
                aria-label={t.close}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {!isZero && (
            <section className="scenario-section" aria-label={t.scenarioTitle}>
              <div className="scenario-heading-row">
                <h2 className="scenario-heading">{t.scenarioTitle}</h2>
                <label className="scenario-picker">
                  <span>{t.allComparisons}</span>
                  <select
                    aria-label={t.allComparisons}
                    value={session.compare ? session.scenario : ""}
                    onChange={(event) =>
                      chooseScenario(event.target.value as ScenarioId)
                    }
                  >
                    <option value="" disabled>
                      {t.chooseComparison}
                    </option>
                    {scenarioIds.map((id) => (
                      <option key={id} value={id}>
                        {t.scenarios[id].title}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="scenario-grid">
                {scenarioIds.map((id) => {
                  const Icon = scenarioIcons[id];
                  return (
                    <button
                      key={id}
                      className={`scenario-button ${session.compare && session.scenario === id ? "selected" : ""}`}
                      onClick={() => chooseScenario(id)}
                      aria-pressed={session.compare && session.scenario === id}
                    >
                      <Icon size={19} />
                      <div>
                        <strong>{t.scenarios[id].title}</strong>
                        <span>{t.scenarios[id].detail}</span>
                      </div>
                      <ChevronRight size={15} />
                    </button>
                  );
                })}
              </div>
            </section>
          )}
          {session.scenario === "zero" && (
            <ZeroExplorer
              config={session.config}
              pair={session.zeroPair}
              onPairChange={(zeroPair) =>
                updateSession({ ...session, zeroPair })
              }
              onPrecisionChange={(precision) =>
                updateSession({
                  ...session,
                  config: { ...session.config, precision },
                })
              }
            />
          )}
          <section
            className={`configuration-strip ${isZero ? "zero-configuration" : ""}`}
            aria-label={t.configure}
          >
            {!isZero && (
              <Field label={t.strategy}>
                <div className="strategy-toggle">
                  {strategies.map((strategy) => (
                    <button
                      key={strategy}
                      aria-label={t.strategies[strategy]}
                      aria-pressed={
                        !session.compare && session.config.strategy === strategy
                      }
                      className={
                        !session.compare && session.config.strategy === strategy
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        updateSession(
                          standardSession(false, "strategies", strategy),
                        )
                      }
                    >
                      {t.strategies[strategy]}
                    </button>
                  ))}
                </div>
              </Field>
            )}
            <Field label={t.ranks}>
              <select
                value={session.config.ranks}
                onChange={(event) => {
                  const ranks = Number(event.target.value) as Config["ranks"];
                  updateSession({
                    ...session,
                    config: {
                      ...session.config,
                      ranks,
                      slowRank:
                        session.config.slowRank !== null &&
                        session.config.slowRank >= ranks
                          ? null
                          : session.config.slowRank,
                    },
                  });
                }}
              >
                {(session.config.strategy === "pp"
                  ? [1, 2, 4]
                  : [1, 2, 4, 8]
                ).map((ranks) => (
                  <option key={ranks} value={ranks}>
                    {t.rankOption(ranks)}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={t.globalBatch}>
              <select
                value={session.config.globalBatch}
                onChange={(event) =>
                  updateSession({
                    ...session,
                    config: {
                      ...session.config,
                      globalBatch: Number(event.target.value),
                    },
                  })
                }
              >
                {Array.from(
                  new Set([session.config.globalBatch, 8, 16, 32, 64, 128]),
                )
                  .sort((a, b) => a - b)
                  .map((batch) => (
                    <option value={batch} key={batch}>
                      {t.sampleOption(batch)}
                    </option>
                  ))}
              </select>
            </Field>
            <div className="batch-identity" title={t.globalFormula}>
              <span>
                <Check size={12} />
                {t.fixedBatch}
              </span>
              <strong>{batchDescription(configs[0])}</strong>
              <small>
                {t.precisionProfiles[session.config.precision]} ·{" "}
                {configs[0].strategy === "pp"
                  ? "GPipe"
                  : configs[0].strategy === "ep"
                    ? t.fixedRouter
                    : t.accumulation}
              </small>
            </div>
            <button
              className="configure-button"
              onClick={() => setSettings(true)}
            >
              <SlidersHorizontal size={15} />
              {t.configure}
            </button>
          </section>
          {!session.compare && (
            <StrategyOverview
              config={session.config}
              onConfigure={() => setSettings(true)}
            />
          )}
          {session.compare && session.scenario !== "zero" && (
            <div className="comparison-note">
              <ArrowLeftRight size={18} />
              <div>
                <strong>{t.scenarios[session.scenario].title}</strong>
                <p>{t.scenarios[session.scenario].note}</p>
                <span>
                  {t.differences} ·{" "}
                  {Object.keys(configs[0])
                    .filter(
                      (key) =>
                        configs[0][key as keyof Config] !==
                        configs[1][key as keyof Config],
                    )
                    .map(
                      (key) =>
                        `${configLabels[key as keyof Config] ?? key} ${String(configs[0][key as keyof Config])} → ${String(configs[1][key as keyof Config])}`,
                    )
                    .join(" · ") || t.noDifferences}
                </span>
              </div>
              <button
                className="icon-button"
                aria-label={t.close}
                onClick={() => updateSession(standardSession(false))}
              >
                <X size={16} />
              </button>
            </div>
          )}
          <section className="playback" aria-label={t.scrub}>
            <div className="playback-controls">
              <button
                className="icon-button"
                aria-label={t.reset}
                title={`${t.reset} · R`}
                onClick={reset}
              >
                <RotateCcw size={16} />
              </button>
              <button
                className="icon-button"
                aria-label={t.previous}
                title={t.previous}
                onClick={() => step(-1)}
                disabled={snapshots[activeRun].eventIndex === 0}
              >
                <SkipBack size={18} />
              </button>
              <button className="play-button" onClick={togglePlay}>
                {playing ? <Pause size={15} /> : <Play size={15} />}
                <span>{playing ? t.pause : t.play}</span>
              </button>
              <button
                className="icon-button"
                aria-label={t.next}
                title={t.next}
                onClick={() => step(1)}
                disabled={
                  snapshots[activeRun].eventIndex ===
                  runs[activeRun].events.length - 1
                }
              >
                <SkipForward size={18} />
              </button>
              <select
                aria-label={t.speed}
                value={speed}
                onChange={(event) => setSpeed(Number(event.target.value))}
              >
                {[0.25, 0.5, 1, 2, 4].map((value) => (
                  <option value={value} key={value}>
                    {value}×
                  </option>
                ))}
              </select>
            </div>
            <div className="scrubber">
              <input
                aria-label={t.scrub}
                type="range"
                min={0}
                max={maxTime}
                step={maxTime / 2000}
                value={time}
                onChange={(event) => seek(Number(event.target.value))}
              />
              <div>
                <span>
                  {t.event}{" "}
                  <strong>
                    {String(snapshots[activeRun].eventIndex).padStart(2, "0")}
                  </strong>{" "}
                  / {runs[activeRun].events.length - 1}
                </span>
                <span>
                  <strong>{ms(time)}</strong> / {ms(maxTime)} {t.modeledTime}
                </span>
              </div>
            </div>
            {session.compare ? (
              <label className="focus-run">
                <span>{t.focusRun}</span>
                <select
                  value={activeRun}
                  onChange={(event) =>
                    setFocusedRun(Number(event.target.value))
                  }
                >
                  <option value={0}>{t.runA}</option>
                  <option value={1}>{t.runB}</option>
                </select>
              </label>
            ) : (
              <span className="keyboard-hint">{t.shortcuts}</span>
            )}
          </section>
          {session.compare && (
            <p className="comparison-hint">{t.inspectHint}</p>
          )}
          <div className={session.compare ? "comparison-grid" : "single-run"}>
            {runs.map((run, index) => (
              <SimulationView
                key={index}
                run={run}
                numerical={numerical[index]}
                snapshot={snapshots[index]}
                time={time}
                maxTime={maxTime}
                focus={focuses[index]}
                onFocus={(focus) => {
                  setFocuses((previous) =>
                    previous.map((value, i) => (i === index ? focus : value)),
                  );
                  setFocusedRun(index);
                }}
                onSeek={seek}
                onBoundary={(eventIndex) => boundary(index, eventIndex)}
                label={index === 0 ? t.runA : t.runB}
                compact={session.compare}
                zeroMode={isZero}
              />
            ))}
          </div>
          <footer>
            <span>
              <span className="status-dot" />
              {t.footer}
            </span>
            <button onClick={() => setGuide(true)}>
              {t.guide}
              <ExternalLink size={12} />
            </button>
          </footer>
        </main>
      </div>
      {settings && (
        <Settings
          config={session.config}
          zeroPair={session.scenario === "zero" ? session.zeroPair : undefined}
          onClose={() => setSettings(false)}
          onApply={(config) => {
            updateSession({ ...session, config });
            setSettings(false);
          }}
        />
      )}
      {guide && (
        <Modal title={t.assumptionsTitle} onClose={() => setGuide(false)} wide>
          <div className="guide-content">
            <p>{t.assumptionsIntro}</p>
            <ol>
              {t.assumptions.map((assumption) => (
                <li key={assumption}>{assumption}</li>
              ))}
            </ol>
            <p className="batch-formula">{t.globalFormula}</p>
            <div className="guide-links">
              <a
                href="https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html"
                target="_blank"
                rel="noreferrer"
              >
                {t.references.fsdp} <ExternalLink size={13} />
              </a>
              <a
                href="https://docs.pytorch.org/docs/stable/notes/ddp.html"
                target="_blank"
                rel="noreferrer"
              >
                {t.references.ddp} <ExternalLink size={13} />
              </a>
              <a
                href="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md"
                target="_blank"
                rel="noreferrer"
              >
                {t.references.nccl} <ExternalLink size={13} />
              </a>
            </div>
          </div>
        </Modal>
      )}
      {shareUrl && (
        <Modal title={t.copyFallback} onClose={() => setShareUrl("")}>
          <textarea
            className="share-input"
            readOnly
            value={shareUrl}
            onFocus={(event) => event.target.select()}
          />
        </Modal>
      )}
    </div>
  );
}

function Settings({
  config,
  zeroPair,
  onApply,
  onClose,
}: {
  config: Config;
  zeroPair?: [ZeroStrategy, ZeroStrategy];
  onApply: (config: Config) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState(config);
  const [error, setError] = useState("");
  const numeric = (
    key: keyof Config,
    label: string,
    min: number,
    max: number,
    step: number | string = 1,
    hint?: string,
  ) => (
    <Field label={label} hint={hint}>
      <input
        type="number"
        required
        min={min}
        max={max}
        step={step}
        value={String(draft[key] ?? "")}
        onChange={(event) =>
          setDraft({
            ...draft,
            [key]:
              event.target.value === ""
                ? Number.NaN
                : Number(event.target.value),
          })
        }
      />
    </Field>
  );
  const apply = () => {
    try {
      onApply(parseConfig(draft));
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
    }
  };
  return (
    <Modal title={t.settingsTitle} onClose={onClose} wide>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          apply();
        }}
      >
        <p className="settings-note">{t.settingsNote}</p>
        <div className="settings-grid">
          {numeric("globalBatch", t.globalBatch, 1, 128)}
          {numeric("modelScale", t.modelScale, 1, 16, 1, t.modelScaleHint)}
          {numeric("seed", t.seed, 0, 4294967295)}
          {numeric("learningRate", t.learningRate, 0.000001, 1, "any")}
        </div>
        {(zeroPair
          ? zeroPair.some((strategy) =>
              ["ddp", "zero1", "zero2"].includes(strategy),
            )
          : draft.strategy === "ddp") && (
          <fieldset>
            <legend>
              {zeroPair
                ? "DDP / ZeRO-1 / ZeRO-2 gradient reduction"
                : t.ddpOnly}
            </legend>
            <div className="settings-grid">
              <Field label={t.bucketLayers}>
                <select
                  value={draft.bucketLayers}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      bucketLayers: Number(
                        event.target.value,
                      ) as Config["bucketLayers"],
                    })
                  }
                >
                  {[1, 2, 4].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </Field>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={draft.overlap}
                  onChange={(event) =>
                    setDraft({ ...draft, overlap: event.target.checked })
                  }
                />
                {t.overlap}
              </label>
            </div>
          </fieldset>
        )}
        {(zeroPair?.includes("zero3") || draft.strategy === "fsdp") && (
          <fieldset>
            <legend>
              {zeroPair ? "ZeRO-3 parameter gathering" : t.fsdpOnly}
            </legend>
            <div className="settings-grid">
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={draft.reshardAfterForward}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      reshardAfterForward: event.target.checked,
                      prefetch: event.target.checked ? draft.prefetch : 0,
                    })
                  }
                />
                {t.reshard}
              </label>
              <Field
                label={t.prefetch}
                hint={!draft.reshardAfterForward ? t.retainedHint : undefined}
              >
                <select
                  value={draft.reshardAfterForward ? draft.prefetch : 0}
                  disabled={!draft.reshardAfterForward}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      prefetch: Number(event.target.value) as 0 | 1,
                    })
                  }
                >
                  <option value={0}>{t.noPrefetch}</option>
                  <option value={1}>{t.onePrefetch}</option>
                </select>
              </Field>
            </div>
          </fieldset>
        )}
        {draft.strategy === "pp" && (
          <fieldset>
            <legend>{t.gpipeTitle}</legend>
            <Field label={t.microbatches} hint={t.ppRankNote}>
              <select
                value={draft.microbatches}
                onChange={(event) =>
                  setDraft({
                    ...draft,
                    microbatches: Number(
                      event.target.value,
                    ) as Config["microbatches"],
                  })
                }
              >
                {[1, 2, 4, 8].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </Field>
          </fieldset>
        )}
        <div className="settings-grid resources">
          {numeric("bandwidthGBps", t.bandwidth, 0.1, 1000, "any")}
          {numeric("latencyUs", t.latency, 0, 1000, "any")}
          {numeric("computeTFLOPS", t.compute, 0.1, 1000, "any")}
          {numeric("capacityMiB", t.capacity, 1, 1048576, "any")}
          <Field label={t.slowRank}>
            <select
              value={draft.slowRank ?? "none"}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  slowRank:
                    event.target.value === "none"
                      ? null
                      : Number(event.target.value),
                })
              }
            >
              <option value="none">{t.noSlowRank}</option>
              {Array.from({ length: draft.ranks }, (_, r) => (
                <option value={r} key={r}>
                  {t.rank(r)}
                </option>
              ))}
            </select>
          </Field>
          {numeric("slowMultiplier", t.multiplier, 1, 10, "any")}
        </div>
        {error && (
          <p className="error-text" role="alert">
            {error}
          </p>
        )}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            {t.cancel}
          </button>
          <button type="submit" className="primary-button">
            {t.apply}
          </button>
        </div>
      </form>
    </Modal>
  );
}
