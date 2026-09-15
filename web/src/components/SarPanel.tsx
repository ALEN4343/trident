import type { SARAnalysis, SARStatus } from "../types";

interface Props {
  status: SARStatus | null;
  result: SARAnalysis | null;
  threshold: number;
  onThreshold: (value: number) => void;
}

export default function SarPanel({ status, result, threshold, onThreshold }: Props) {
  return (
    <>
      <section className="panel">
        <div className="panel-head">
          Oil segmenter
          {status && (
            <span className={`chip ${status.available ? "ok" : "mute"}`}>
              {status.available ? "loaded" : "no weights"}
            </span>
          )}
        </div>
        <div className="panel-body">
          <p className="hint" style={{ marginBottom: 11 }}>
            DeepLabv3+ trained on Sentinel-1 and PALSAR backscatter. This reads how oil
            flattens the sea surface and kills radar return — a different physical signal
            from the optical branch, not a second opinion on it.
          </p>

          {status && !status.available && (
            <div className="banner warn" style={{ marginBottom: 11 }}>
              No checkpoint: {status.reason}. Train one with{" "}
              <code>scripts/train_oil.py</code>.
            </div>
          )}

          {status?.available && status.representative === false && (
            <div className="banner warn" style={{ marginBottom: 11 }}>
              These weights come from a reduced or CPU smoke-test run. The mask proves the
              pipeline executes; it is not a measurement.
            </div>
          )}

          {status?.available && (
            <div className="stat-strip" style={{ marginTop: 0 }}>
              <div className="stat">
                <div className="k">Epoch</div>
                <div className="v">{status.epoch ?? "—"}</div>
              </div>
              <div className="stat">
                <div className="k">Oil IoU</div>
                <div className="v">
                  {status.oil_iou != null ? status.oil_iou.toFixed(3) : "—"}
                </div>
              </div>
              <div className="stat">
                <div className="k">Size</div>
                <div className="v">{status.size_mb ?? "—"}MB</div>
              </div>
            </div>
          )}

          <div className="field" style={{ marginTop: 13, marginBottom: 0 }}>
            <label>
              Decision threshold — {threshold.toFixed(2)}
              <span style={{ color: "var(--dimmer)" }}> (lower finds more, costs precision)</span>
            </label>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(e) => onThreshold(Number(e.target.value))}
            />
          </div>
        </div>
      </section>

      {result && (
        <section className="panel">
          <div className="severity">
            <div className="severity-top">
              <div className="severity-score" style={{ color: result.severity.colour }}>
                {result.severity.score.toFixed(0)}
              </div>
              <div className="severity-meta">
                <div className="severity-band" style={{ color: result.severity.colour }}>
                  {result.severity.band}
                </div>
                <div className="severity-range">
                  MPSI from surface coverage alone
                </div>
              </div>
            </div>

            <div className="stat-strip">
              <div className="stat">
                <div className="k">Coverage</div>
                <div className="v">{result.coverage_percent.toFixed(2)}%</div>
              </div>
              <div className="stat">
                <div className="k">Slicks</div>
                <div className="v">{result.slick_count}</div>
              </div>
              <div className="stat">
                <div className="k">Threshold</div>
                <div className="v">{result.threshold.toFixed(2)}</div>
              </div>
            </div>

            {result.model.caveat && (
              <div className="banner warn" style={{ marginTop: 11, marginBottom: 0 }}>
                {result.model.caveat}
              </div>
            )}
          </div>
        </section>
      )}
    </>
  );
}
