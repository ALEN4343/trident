import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { DeclaredPose } from "./api";
import type { Analysis, Drift, Mission, Telemetry } from "./types";
import ScenePanel from "./components/ScenePanel";
import AnalysisPanel from "./components/AnalysisPanel";
import MissionPanel from "./components/MissionPanel";
import MapView from "./components/MapView";
import ImageryView from "./components/ImageryView";

const STAGES = ["DETECT", "LOCALISE", "ATTRIBUTE", "FORECAST", "DISPATCH", "VERIFY"] as const;

const INITIAL_POSE: DeclaredPose = {
  lat: 9.9658,
  lon: 76.2422,
  altitude_m: 80,
  pitch_deg: -90,
  yaw_deg: 0,
  hfov_deg: 84,
};

export default function App() {
  const [pose, setPose] = useState<DeclaredPose>(INITIAL_POSE);
  const [sensitivity, setSensitivity] = useState(0.3);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<"map" | "imagery">("imagery");

  const [drift, setDrift] = useState<Drift | null>(null);
  const [reverseDrift, setReverseDrift] = useState<Drift | null>(null);
  const [driftBusy, setDriftBusy] = useState(false);

  const [mission, setMission] = useState<Mission | null>(null);
  const [planning, setPlanning] = useState(false);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [running, setRunning] = useState(false);
  const [complete, setComplete] = useState<Record<string, number> | null>(null);
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => () => socket.current?.close(), []);

  const reset = () => {
    setAnalysis(null);
    setMission(null);
    setDrift(null);
    setReverseDrift(null);
    setTelemetry(null);
    setComplete(null);
    setSelected(null);
    setError(null);
  };

  const onFile = useCallback(
    async (chosen: File) => {
      reset();
      setFile(chosen);
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(chosen);
      });
      setBusy(true);
      try {
        const result = await api.analyse(chosen, pose, sensitivity);
        setAnalysis(result);
        setView("imagery");
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [pose, sensitivity],
  );

  const reanalyse = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await api.analyse(file, pose, sensitivity));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [file, pose, sensitivity]);

  const selectedItem = useMemo(
    () => analysis?.items.find((i) => i.id === selected) ?? null,
    [analysis, selected],
  );

  const runDrift = useCallback(
    async (reverse: boolean) => {
      const target =
        selectedItem ?? analysis?.items.find((i) => i.lat !== null) ?? null;
      const lat = target?.lat ?? analysis?.capture.pose?.lat;
      const lon = target?.lon ?? analysis?.capture.pose?.lon;
      if (lat == null || lon == null) return;

      setDriftBusy(true);
      setError(null);
      try {
        const result = await api.drift(
          lat,
          lon,
          target?.class_id ?? "plastic_bottle",
          reverse ? 48 : 6,
          reverse,
        );
        if (reverse) setReverseDrift(result);
        else setDrift(result);
        setView("map");
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setDriftBusy(false);
      }
    },
    [analysis, selectedItem],
  );

  const planMission = useCallback(async () => {
    if (!analysis) return;
    setPlanning(true);
    setError(null);
    try {
      setMission(
        await api.planMission({
          session_id: analysis.session_id,
          base_lat: pose.lat - 0.012,
          base_lon: pose.lon - 0.012,
          drift_aware: true,
          cruise_speed_ms: 1.6,
          payload_capacity_kg: 180,
          endurance_min: 240,
        }),
      );
      setView("map");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPlanning(false);
    }
  }, [analysis, pose]);

  const onGate = useCallback(
    async (index: number, decision: "approve" | "skip" | "flag") => {
      if (!mission) return;
      try {
        setMission(await api.resolveGate(mission.id, index, decision));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [mission],
  );

  const launch = useCallback(() => {
    if (!mission) return;
    setRunning(true);
    setComplete(null);
    setView("map");

    const ws = api.missionSocket(mission.id);
    socket.current = ws;

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "telemetry") setTelemetry(message as Telemetry);
      else if (message.type === "complete") {
        setComplete(message);
        setRunning(false);
      } else if (message.type === "error") {
        setError(message.message);
        setRunning(false);
      }
    };
    ws.onerror = () => {
      setError("Telemetry socket failed");
      setRunning(false);
    };
    ws.onclose = () => setRunning(false);
  }, [mission]);

  const stageOn = (stage: string) => {
    switch (stage) {
      case "DETECT":
        return !!analysis;
      case "LOCALISE":
        return !!analysis?.georeferenced;
      case "ATTRIBUTE":
        return !!reverseDrift;
      case "FORECAST":
        return !!drift;
      case "DISPATCH":
        return !!mission;
      case "VERIFY":
        return !!complete;
      default:
        return false;
    }
  };

  const hasLocated = !!analysis?.items.some((i) => i.lat !== null);

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          TRI<span>DENT</span>
          <small>Marine Pollution Intelligence</small>
        </div>
        <nav className="pipeline">
          {STAGES.map((stage, i) => (
            <span key={stage} style={{ display: "contents" }}>
              {i > 0 && <span className="stage-sep">›</span>}
              <span className={`stage ${stageOn(stage) ? "on" : ""}`}>{stage}</span>
            </span>
          ))}
        </nav>
      </header>

      <div className="columns">
        <aside className="rail left">
          <ScenePanel
            pose={pose}
            onPose={setPose}
            sensitivity={sensitivity}
            onSensitivity={setSensitivity}
            onFile={onFile}
            busy={busy}
            analysis={analysis}
            fileName={file?.name ?? null}
          />
          {analysis && (
            <section className="panel">
              <div className="panel-body" style={{ paddingTop: 13 }}>
                <button className="btn ghost" onClick={reanalyse} disabled={busy}>
                  Re-analyse with this geometry
                </button>
              </div>
            </section>
          )}
        </aside>

        <main className="stage-area">
          {analysis && preview ? (
            <>
              <div className="view-toggle">
                <button className={view === "imagery" ? "on" : ""} onClick={() => setView("imagery")}>
                  IMAGERY
                </button>
                <button className={view === "map" ? "on" : ""} onClick={() => setView("map")}>
                  MAP
                </button>
              </div>

              {view === "imagery" ? (
                <ImageryView
                  src={preview}
                  analysis={analysis}
                  selected={selected}
                  onSelect={setSelected}
                  showRejections
                />
              ) : (
                <MapView
                  analysis={analysis}
                  drift={drift}
                  reverseDrift={reverseDrift}
                  mission={mission}
                  telemetry={telemetry}
                  selected={selected}
                  onSelect={setSelected}
                />
              )}

              {view === "map" && telemetry && (
                <div className="hud">
                  <div className="hud-state">
                    <span className="pulse" />
                    {telemetry.state.replace("_", " ")}
                  </div>
                  <div className="hud-grid">
                    <div>
                      <div className="k">Battery</div>
                      <div className="v">{telemetry.battery_pct.toFixed(0)}%</div>
                    </div>
                    <div>
                      <div className="k">Payload</div>
                      <div className="v">{telemetry.payload_kg.toFixed(1)} kg</div>
                    </div>
                    <div>
                      <div className="k">Heading</div>
                      <div className="v">{telemetry.heading_deg.toFixed(0)}°</div>
                    </div>
                    <div>
                      <div className="k">Elapsed</div>
                      <div className="v">{(telemetry.t / 60).toFixed(0)} min</div>
                    </div>
                  </div>
                  {telemetry.message && <div className="hud-msg">{telemetry.message}</div>}
                </div>
              )}
            </>
          ) : (
            <div className="empty">
              <h2>No scene loaded</h2>
              <p>
                Drop a water body image on the left. TRIDENT will classify pollution, place each
                detection on the map, forecast where it drifts, and plan a cleanup route you
                authorise.
              </p>
            </div>
          )}
        </main>

        <aside className="rail right">
          {error && (
            <section className="panel">
              <div className="panel-body" style={{ paddingTop: 13 }}>
                <div className="banner warn" style={{ marginBottom: 0 }}>
                  {error}
                </div>
              </div>
            </section>
          )}

          {analysis ? (
            <>
              <AnalysisPanel analysis={analysis} selected={selected} onSelect={setSelected} />

              <section className="panel">
                <div className="panel-head">
                  Drift
                  {drift?.synthetic_forcing && <span className="chip warn">synthetic forcing</span>}
                </div>
                <div className="panel-body">
                  <p className="hint" style={{ marginBottom: 11 }}>
                    {selectedItem
                      ? `Using ${selectedItem.label} — windage from its buoyancy class.`
                      : "Select a detection to drift it, or use the first georeferenced item."}
                  </p>
                  <div className="grid-2">
                    <button className="btn" onClick={() => runDrift(false)} disabled={!hasLocated || driftBusy}>
                      Forecast 6 h
                    </button>
                    <button className="btn ghost" onClick={() => runDrift(true)} disabled={!hasLocated || driftBusy}>
                      Trace source
                    </button>
                  </div>

                  {drift && (
                    <div className="stat-strip">
                      <div className="stat">
                        <div className="k">Drift 6h</div>
                        <div className="v">{(drift.displacement_m / 1000).toFixed(2)}km</div>
                      </div>
                      <div className="stat">
                        <div className="k">Spread</div>
                        <div className="v">{drift.spread_m.toFixed(0)}m</div>
                      </div>
                      <div className="stat">
                        <div className="k">Windage</div>
                        <div className="v">{(drift.windage * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  )}

                  {reverseDrift && (
                    <div className="banner info" style={{ marginTop: 11, marginBottom: 0 }}>
                      Hindcast places the likely entry point{" "}
                      <b>{(reverseDrift.displacement_m / 1000).toFixed(1)} km</b> upstream over 48 h.
                    </div>
                  )}
                </div>
              </section>

              <MissionPanel
                mission={mission}
                planning={planning}
                canPlan={analysis.collectable_targets > 0}
                running={running}
                complete={complete}
                telemetry={telemetry}
                onPlan={planMission}
                onGate={onGate}
                onLaunch={launch}
              />
            </>
          ) : (
            <section className="panel">
              <div className="panel-head">Awaiting scene</div>
              <div className="panel-body">
                <p className="hint">
                  Severity, rejected candidates, drift and the cleanup mission appear here once an
                  image is analysed.
                </p>
              </div>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
