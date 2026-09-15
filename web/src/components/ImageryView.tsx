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
  src: string;
  analysis: Analysis;
  selected: string | null;
  onSelect: (id: string | null) => void;
  showRejections: boolean;
}

export default function ImageryView({ src, analysis, selected, onSelect, showRejections }: Props) {
  const { width, height } = analysis.image;

  return (
    <div className="imagery">
      <div className="imagery-inner">
        <img src={src} alt="Analysed scene" />
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
          {/* Accepted amorphous regions */}
          {analysis.regions.map((region) => (
            <polygon
              key={region.id}
              points={region.polygon_px.map((p) => p.join(",")).join(" ")}
              fill="#facc1526"
              stroke="#facc15"
              strokeWidth={Math.max(2, width / 400)}
            />
          ))}

          {/* Candidates that were considered and thrown out */}
          {showRejections &&
            analysis.rejections.map((rejection) => (
              <polygon
                key={rejection.id}
                points={rejection.polygon_px.map((p) => p.join(",")).join(" ")}
                fill="none"
                stroke="#55697e"
                strokeWidth={Math.max(1.5, width / 600)}
                strokeDasharray={`${width / 90} ${width / 140}`}
              />
            ))}

          {/* Countable items */}
          {analysis.items.map((item) => {
            const [x1, y1, x2, y2] = item.bbox;
            const colour = GROUP_COLOUR[item.group] ?? "#22d3ee";
            const isOn = selected === item.id;
            return (
              <g
                key={item.id}
                onClick={() => onSelect(isOn ? null : item.id)}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill={isOn ? `${colour}33` : "transparent"}
                  stroke={colour}
                  strokeWidth={Math.max(2, width / (isOn ? 300 : 450))}
                  rx={2}
                />
                <rect
                  x={x1}
                  y={Math.max(0, y1 - height / 34)}
                  width={Math.max(item.label.length * (width / 78), width / 11)}
                  height={height / 34}
                  fill={colour}
                  rx={1.5}
                />
                <text
                  x={x1 + width / 200}
                  y={Math.max(0, y1 - height / 34) + height / 48}
                  fontSize={height / 46}
                  fontFamily="ui-monospace, monospace"
                  fontWeight="700"
                  fill="#04131a"
                >
                  {item.label} {(item.confidence * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="legend">
        <div className="legend-title">Overlay</div>
        <div className="legend-row">
          <span className="legend-key" style={{ background: "#22d3ee" }} />
          Countable debris
        </div>
        <div className="legend-row">
          <span className="legend-key" style={{ background: "#facc15" }} />
          Surface contaminant
        </div>
        {showRejections && (
          <div className="legend-row">
            <span
              className="legend-key"
              style={{ background: "repeating-linear-gradient(90deg,#55697e 0 4px,transparent 4px 7px)" }}
            />
            Rejected candidate
          </div>
        )}
      </div>
    </div>
  );
}
