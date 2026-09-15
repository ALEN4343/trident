import { useRef, useState } from "react";
import type { DeclaredPose } from "../api";
import type { Analysis } from "../types";

const PRESETS: Record<string, { lat: number; lon: number }> = {
  "Kochi harbour mouth": { lat: 9.9658, lon: 76.2422 },
  "Mumbai harbour": { lat: 18.9435, lon: 72.8365 },
  "Chennai Marina": { lat: 13.0524, lon: 80.2824 },
  "Bellandur lake": { lat: 12.9366, lon: 77.6664 },
};

const CAPTURE_LABEL: Record<Analysis["capture"]["source"], string> = {
  exif_full: "Drone telemetry",
  exif_gps: "EXIF position only",
  declared: "Operator declared",
  none: "No position",
};

interface Props {
  pose: DeclaredPose;
  onPose: (pose: DeclaredPose) => void;
  sensitivity: number;
  onSensitivity: (value: number) => void;
  onFile: (file: File) => void;
  busy: boolean;
  analysis: Analysis | null;
  fileName: string | null;
}

export default function ScenePanel({
  pose,
  onPose,
  sensitivity,
  onSensitivity,
  onFile,
  busy,
  analysis,
  fileName,
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const set = (key: keyof DeclaredPose) => (event: React.ChangeEvent<HTMLInputElement>) =>
    onPose({ ...pose, [key]: Number(event.target.value) });

  return (
    <>
      <section className="panel">
        <div className="panel-head">Scene</div>
        <div className="panel-body">
          <div
            className={`dropzone ${over ? "over" : ""}`}
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file) onFile(file);
            }}
          >
            {busy ? (
              <>
                <span className="spinner" />
                <strong style={{ marginTop: 8 }}>Analysing</strong>
              </>
            ) : (
              <>
                <strong>{fileName ?? "Drop a water body image"}</strong>
                {fileName ? "click to replace" : "or click to browse"}
              </>
            )}
          </div>
          <input
            ref={input}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFile(file);
            }}
          />
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          Capture geometry
          {analysis && (
            <span className={`chip ${analysis.capture.source === "exif_full" ? "ok" : "mute"}`}>
              {CAPTURE_LABEL[analysis.capture.source]}
            </span>
          )}
        </div>
        <div className="panel-body">
          <p className="hint" style={{ marginTop: 0, marginBottom: 11 }}>
            Read from EXIF when the image carries it. Dataset photos do not, so declare the
            capture pose here — every coordinate downstream inherits this assumption.
          </p>

          <div className="field">
            <label>Location preset</label>
            <select
              onChange={(e) => {
                const preset = PRESETS[e.target.value];
                if (preset) onPose({ ...pose, ...preset });
              }}
              defaultValue=""
            >
              <option value="">Choose…</option>
              {Object.keys(PRESETS).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div className="grid-2">
            <div className="field">
              <label>Latitude</label>
              <input type="number" step="0.0001" value={pose.lat} onChange={set("lat")} />
            </div>
            <div className="field">
              <label>Longitude</label>
              <input type="number" step="0.0001" value={pose.lon} onChange={set("lon")} />
            </div>
          </div>

          <div className="grid-2">
            <div className="field">
              <label>Altitude AGL (m)</label>
              <input type="number" step="1" value={pose.altitude_m} onChange={set("altitude_m")} />
            </div>
            <div className="field">
              <label>Camera HFOV (°)</label>
              <input type="number" step="1" value={pose.hfov_deg} onChange={set("hfov_deg")} />
            </div>
          </div>

          <div className="grid-2">
            <div className="field">
              <label>Gimbal pitch (°)</label>
              <input type="number" step="1" value={pose.pitch_deg} onChange={set("pitch_deg")} />
            </div>
            <div className="field">
              <label>Heading (°)</label>
              <input type="number" step="1" value={pose.yaw_deg} onChange={set("yaw_deg")} />
            </div>
          </div>

          <div className="field">
            <label>
              Ecological sensitivity — {(sensitivity * 100).toFixed(0)}%
              <span style={{ color: "var(--dimmer)" }}> (protected habitat proximity)</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={sensitivity}
              onChange={(e) => onSensitivity(Number(e.target.value))}
            />
          </div>
        </div>
      </section>

      {analysis && (
        <section className="panel">
          <div className="panel-head">Frame</div>
          <div className="panel-body">
            {analysis.capture.warnings.map((warning) => (
              <div className="banner warn" key={warning}>
                {warning}
              </div>
            ))}
            {analysis.notes.map((note) => (
              <div className="banner info" key={note}>
                {note}
              </div>
            ))}
            <div className="stat-strip">
              <div className="stat">
                <div className="k">Water</div>
                <div className="v">{(analysis.water_fraction * 100).toFixed(0)}%</div>
              </div>
              <div className="stat">
                <div className="k">Imaged area</div>
                <div className="v">
                  {analysis.water_area_m2
                    ? analysis.water_area_m2 > 9999
                      ? `${(analysis.water_area_m2 / 10000).toFixed(1)}ha`
                      : `${Math.round(analysis.water_area_m2)}m²`
                    : "—"}
                </div>
              </div>
              <div className="stat">
                <div className="k">Mappable</div>
                <div className="v">{analysis.collectable_targets}</div>
              </div>
            </div>
          </div>
        </section>
      )}
    </>
  );
}
