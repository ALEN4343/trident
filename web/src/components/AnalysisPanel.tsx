import type { Analysis } from "../types";

const GROUP_COLOUR: Record<string, string> = {
  macroplastic: "#22d3ee",
  ghost_gear: "#f43f5e",
  metal_glass: "#a78bfa",
  rubber: "#fb923c",
  organic_waste: "#84cc16",
  surface_film: "#facc15",
};

interface Props {
  analysis: Analysis;
  selected: string | null;
  onSelect: (id: string | null) => void;
}

export default function AnalysisPanel({ analysis, selected, onSelect }: Props) {
  const { severity } = analysis;
  const maxContribution = Math.max(...severity.components.map((c) => c.weight * 100), 1);

  return (
    <>
      <section className="panel">
        <div className="severity">
          <div className="severity-top">
            <div className="severity-score" style={{ color: severity.colour }}>
              {severity.score.toFixed(0)}
            </div>
            <div className="severity-meta">
              <div className="severity-band" style={{ color: severity.colour }}>
                {severity.band}
              </div>
              <div className="severity-range">
                MPSI · confidence band {severity.lower.toFixed(0)}–{severity.upper.toFixed(0)}
              </div>
            </div>
          </div>

          {severity.components.map((component) => (
            <div className="bar-row" key={component.key} title={component.description}>
              <div className="bar-label">
                <span>
                  {component.key.replace("_", " ")}
                  <span style={{ color: "var(--dimmer)" }}> ×{component.weight}</span>
                </span>
                <b>{component.contribution.toFixed(1)}</b>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width: `${(component.contribution / maxContribution) * 100}%`,
                    background: severity.colour,
                  }}
                />
              </div>
            </div>
          ))}

          <div className="stat-strip">
            <div className="stat">
              <div className="k">Items</div>
              <div className="v">{severity.item_count}</div>
            </div>
            <div className="stat">
              <div className="k">Mass</div>
              <div className="v">{severity.total_mass_kg.toFixed(2)}kg</div>
            </div>
            <div className="stat">
              <div className="k">Scrap value</div>
              <div className="v">₹{severity.recoverable_value_inr.toFixed(0)}</div>
            </div>
          </div>

          {severity.notes.map((note) => (
            <div className="banner info" key={note} style={{ marginTop: 11, marginBottom: 0 }}>
              {note}
            </div>
          ))}
        </div>
      </section>

      {analysis.safety.wildlife.length > 0 && (
        <section className="panel">
          <div className="panel-body" style={{ paddingTop: 13 }}>
            <div className="banner warn" style={{ marginBottom: 0 }}>
              <strong>Wildlife in frame:</strong>&nbsp;{analysis.safety.wildlife.join(", ")}.
              Autonomous collection is inhibited for this scene.
            </div>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="panel-head">
          Detections
          <span className="count">{analysis.items.length + analysis.regions.length}</span>
        </div>
        <div className="panel-body">
          {analysis.items.length === 0 && analysis.regions.length === 0 && (
            <p className="hint">No pollution above the confidence floor.</p>
          )}

          {analysis.regions.map((region) => (
            <div className="row" key={region.id}>
              <span className="swatch" style={{ background: "#facc15" }} />
              <div className="row-main">
                <div className="row-title">
                  <b>{region.label}</b>
                  <span className="chip warn">{region.percent_of_water.toFixed(1)}% surface</span>
                </div>
                <div className="row-sub">
                  conf {(region.confidence * 100).toFixed(0)}% · smooth{" "}
                  {region.evidence.smoothness?.toFixed(2)} · irid{" "}
                  {region.evidence.iridescence?.toFixed(2)}
                </div>
              </div>
            </div>
          ))}

          {analysis.items.map((item) => (
            <div
              className={`row clickable ${selected === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => onSelect(selected === item.id ? null : item.id)}
            >
              <span className="swatch" style={{ background: GROUP_COLOUR[item.group] ?? "#22d3ee" }} />
              <div className="row-main">
                <div className="row-title">
                  <b>{item.label}</b>
                  {!item.auto_collect && <span className="chip danger">sign-off</span>}
                  <span className="chip mute">{(item.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="row-sub">
                  {item.lat !== null
                    ? `${item.lat.toFixed(5)}, ${item.lon!.toFixed(5)}`
                    : "not georeferenced"}
                  {item.uncertainty_m !== null && ` ±${item.uncertainty_m.toFixed(0)}m`}
                  {item.size_m !== null && ` · ${(item.size_m * 100).toFixed(0)}cm`}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          Rejected candidates
          <span className="count">{analysis.rejections.length}</span>
        </div>
        <div className="panel-body">
          <p className="hint" style={{ marginBottom: 11 }}>
            Natural features are mistaken for pollution more often than pollution is missed.
            These were considered and ruled out, so the score above is what survived.
          </p>

          {analysis.rejections.length === 0 && (
            <p className="hint" style={{ color: "var(--dimmer)" }}>
              Nothing was proposed and rejected in this frame.
            </p>
          )}

          {analysis.rejections.map((rejection) => (
            <div className="rejected" key={rejection.id}>
              <div className="row-title">
                <b>{rejection.label}</b>
                <span className="chip mute">
                  not {rejection.would_have_been.toLowerCase()}
                </span>
              </div>
              <div className="row-sub">margin {rejection.margin.toFixed(2)}</div>
              <div className="why">{rejection.discriminator}</div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
