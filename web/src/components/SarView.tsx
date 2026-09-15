import type { SARAnalysis } from "../types";

interface Props {
  src: string;
  result: SARAnalysis;
}

export default function SarView({ src, result }: Props) {
  const { width, height } = result.image;
  const stroke = Math.max(2, width / 350);

  return (
    <div className="imagery">
      <div className="imagery-inner">
        <img src={src} alt="SAR scene" />
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
          {result.polygons_px.map((shape, i) => (
            <polygon
              key={i}
              points={shape.map((p) => p.join(",")).join(" ")}
              fill="#f0514f22"
              stroke="#f0514f"
              strokeWidth={stroke}
              strokeLinejoin="round"
            />
          ))}
        </svg>
      </div>

      <div className="legend">
        <div className="legend-title">SAR segmentation</div>
        <div className="legend-row">
          <span className="legend-key" style={{ background: "#f0514f" }} />
          Oil, {result.coverage_percent.toFixed(1)}% of scene
        </div>
        <div className="legend-row">
          <span className="legend-key" style={{ background: "var(--dimmer)" }} />
          {result.slick_count} separate slick{result.slick_count === 1 ? "" : "s"}
        </div>
      </div>
    </div>
  );
}
