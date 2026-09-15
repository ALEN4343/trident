import type { Mission, Telemetry } from "../types";

function duration(seconds: number): string {
  const m = Math.round(seconds / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
}

interface Props {
  mission: Mission | null;
  planning: boolean;
  canPlan: boolean;
  running: boolean;
  complete: Record<string, number> | null;
  telemetry: Telemetry | null;
  onPlan: () => void;
  onGate: (index: number, decision: "approve" | "skip" | "flag") => void;
  onLaunch: () => void;
}

export default function MissionPanel({
  mission,
  planning,
  canPlan,
  running,
  complete,
  telemetry,
  onPlan,
  onGate,
  onLaunch,
}: Props) {
  if (!mission) {
    return (
      <section className="panel">
        <div className="panel-head">Cleanup mission</div>
        <div className="panel-body">
          <p className="hint" style={{ marginBottom: 11 }}>
            Clusters georeferenced debris into hotspots and plans a route against where each
            item is forecast to have drifted by the time the vessel arrives.
          </p>
          <button className="btn" onClick={onPlan} disabled={!canPlan || planning}>
            {planning ? "Planning…" : "Plan mission"}
          </button>
          {!canPlan && (
            <p className="hint" style={{ marginTop: 9, color: "var(--dimmer)" }}>
              Needs at least one georeferenced detection.
            </p>
          )}
        </div>
      </section>
    );
  }

  const blocking = mission.gates.filter((g) => g.blocking && !g.resolved);
  const open = mission.gates.filter((g) => !g.resolved);

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          Mission {mission.id}
          {mission.drift_aware && <span className="chip info">drift-aware</span>}
        </div>
        <div className="panel-body">
          <div className="stat-strip" style={{ marginTop: 0 }}>
            <div className="stat">
              <div className="k">Waypoints</div>
              <div className="v">{mission.summary.waypoints}</div>
            </div>
            <div className="stat">
              <div className="k">Distance</div>
              <div className="v">{(mission.summary.distance_m / 1000).toFixed(2)}km</div>
            </div>
            <div className="stat">
              <div className="k">Duration</div>
              <div className="v">{duration(mission.summary.duration_s)}</div>
            </div>
            <div className="stat">
              <div className="k">Items</div>
              <div className="v">{mission.summary.items}</div>
            </div>
            <div className="stat">
              <div className="k">Payload</div>
              <div className="v">{mission.summary.payload_kg.toFixed(1)}kg</div>
            </div>
            <div className="stat">
              <div className="k">Energy</div>
              <div className="v">{mission.summary.energy_used_pct.toFixed(0)}%</div>
            </div>
          </div>

          {mission.drift_aware && (
            <p className="hint" style={{ marginTop: 11 }}>
              Waypoints are offset from the observed positions by up to{" "}
              <b style={{ color: "var(--cyan)" }}>
                {Math.max(...mission.waypoints.map((w) => w.drift_offset_m), 0).toFixed(0)} m
              </b>{" "}
              to intercept the debris where it is forecast to be on arrival.
            </p>
          )}

          {mission.notes.map((note) => (
            <div className="banner info" key={note} style={{ marginTop: 11, marginBottom: 0 }}>
              {note}
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          Human authorisation
          <span className="count">
            {mission.gates.length - open.length}/{mission.gates.length} cleared
          </span>
        </div>
        <div className="panel-body">
          {mission.gates.map((gate, index) => (
            <div
              className={`gate ${gate.resolved ? "resolved" : gate.blocking ? "blocking" : ""}`}
              key={`${gate.reason}-${index}`}
            >
              <div className="gate-reason">
                {gate.reason}
                {gate.blocking && !gate.resolved && <span className="chip danger">blocking</span>}
                {gate.resolved && <span className="chip ok">{gate.decision}</span>}
              </div>
              <div className="gate-detail">{gate.detail}</div>
              {!gate.resolved && (
                <div className="gate-actions">
                  <button className="approve" onClick={() => onGate(index, "approve")}>
                    APPROVE
                  </button>
                  <button className="skip" onClick={() => onGate(index, "skip")}>
                    SKIP
                  </button>
                  <button className="flag" onClick={() => onGate(index, "flag")}>
                    FLAG
                  </button>
                </div>
              )}
            </div>
          ))}

          <button
            className="btn primary"
            style={{ marginTop: 4 }}
            onClick={onLaunch}
            disabled={blocking.length > 0 || running}
          >
            {running
              ? "Mission under way…"
              : blocking.length > 0
                ? `${blocking.length} gate${blocking.length > 1 ? "s" : ""} to clear`
                : "Launch mission"}
          </button>
        </div>
      </section>

      {complete && (
        <section className="panel">
          <div className="panel-head">Mission report</div>
          <div className="panel-body">
            <div className="stat-strip" style={{ marginTop: 0 }}>
              <div className="stat">
                <div className="k">Visited</div>
                <div className="v">{complete.waypoints_visited}</div>
              </div>
              <div className="stat">
                <div className="k">Skipped</div>
                <div className="v">{complete.waypoints_skipped}</div>
              </div>
              <div className="stat">
                <div className="k">Collected</div>
                <div className="v">{complete.items_collected}</div>
              </div>
              <div className="stat">
                <div className="k">Payload</div>
                <div className="v">{complete.payload_kg.toFixed(1)}kg</div>
              </div>
              <div className="stat">
                <div className="k">Battery</div>
                <div className="v">{complete.battery_remaining_pct.toFixed(0)}%</div>
              </div>
              <div className="stat">
                <div className="k">Elapsed</div>
                <div className="v">{duration(complete.elapsed_s)}</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {telemetry && !complete && (
        <section className="panel">
          <div className="panel-head">Live telemetry</div>
          <div className="panel-body">
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="swatch" style={{ background: "var(--cyan)" }} />
              <div className="row-main">
                <div className="row-title">
                  <b>{telemetry.state.replace("_", " ")}</b>
                  <span className="chip info">wp {telemetry.seq}</span>
                </div>
                <div className="row-sub">
                  {telemetry.lat.toFixed(5)}, {telemetry.lon.toFixed(5)} ·{" "}
                  {telemetry.battery_pct.toFixed(0)}% · {telemetry.payload_kg.toFixed(1)}kg
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </>
  );
}
