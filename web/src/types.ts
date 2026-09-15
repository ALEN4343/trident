export interface Item {
  id: string;
  class_id: string;
  label: string;
  group: string;
  confidence: number;
  bbox: [number, number, number, number];
  centroid_px: [number, number];
  hazard: number;
  mass_kg: number;
  auto_collect: boolean;
  lat: number | null;
  lon: number | null;
  uncertainty_m: number | null;
  size_m: number | null;
}

export interface Region {
  id: string;
  class_id: string;
  label: string;
  confidence: number;
  percent_of_water: number;
  polygon_px: [number, number][];
  evidence: Record<string, number>;
}

export interface Rejection {
  id: string;
  rejected_as: string;
  label: string;
  would_have_been: string;
  margin: number;
  discriminator: string;
  polygon_px: [number, number][];
  evidence: Record<string, number>;
}

export interface SeverityComponent {
  key: string;
  value: number;
  weight: number;
  contribution: number;
  description: string;
}

export interface Severity {
  score: number;
  band: string;
  colour: string;
  lower: number;
  upper: number;
  components: SeverityComponent[];
  item_count: number;
  total_mass_kg: number;
  recoverable_value_inr: number;
  dominant_hazard: string | null;
  notes: string[];
}

export interface Capture {
  source: "exif_full" | "exif_gps" | "declared" | "none";
  georeferenced: boolean;
  has_attitude: boolean;
  pose: {
    lat: number;
    lon: number;
    altitude_m: number;
    yaw_deg: number;
    pitch_deg: number;
    roll_deg: number;
  } | null;
  intrinsics: { width: number; height: number; hfov_deg: number };
  warnings: string[];
}

export interface Analysis {
  session_id: string;
  image: { width: number; height: number };
  georeferenced: boolean;
  water_fraction: number;
  water_area_m2: number | null;
  footprint: [number, number][] | null;
  items: Item[];
  regions: Region[];
  rejections: Rejection[];
  severity: Severity;
  safety: { wildlife: string[]; vessels: number; people: number };
  detector: { model: string; available: boolean; note: string | null };
  capture: Capture;
  collectable_targets: number;
  notes: string[];
}

export interface Gate {
  kind: "mission" | "pickup";
  reason: string;
  detail: string;
  blocking: boolean;
  hotspot_id: string | null;
  target_id: string | null;
  resolved: boolean;
  decision: string | null;
}

export interface Waypoint {
  seq: number;
  hotspot_id: string;
  lat: number;
  lon: number;
  observed_lat: number;
  observed_lon: number;
  drift_offset_m: number;
  leg_distance_m: number;
  eta_s: number;
  payload_after_kg: number;
  energy_used_pct: number;
  item_count: number;
  needs_signoff: boolean;
}

export interface Mission {
  id: string;
  vehicle: { name: string; cruise_speed_ms: number; payload_capacity_kg: number; endurance_s: number };
  base: { lat: number; lon: number };
  drift_aware: boolean;
  waypoints: Waypoint[];
  gates: Gate[];
  deferred: { id: string; lat: number; lon: number; items: number }[];
  summary: {
    waypoints: number;
    items: number;
    distance_m: number;
    duration_s: number;
    payload_kg: number;
    energy_used_pct: number;
    approved: boolean;
    blocking_gates: number;
  };
  notes: string[];
}

export interface Drift {
  geojson: GeoJSON.FeatureCollection;
  track: { lat: number; lon: number }[];
  displacement_m: number;
  spread_m: number;
  windage: number;
  reverse: boolean;
  synthetic_forcing: boolean;
  hours: number;
}

export interface Telemetry {
  type: "telemetry";
  t: number;
  lat: number;
  lon: number;
  heading_deg: number;
  speed_ms: number;
  battery_pct: number;
  payload_kg: number;
  state: string;
  seq: number;
  message: string;
}

export interface SARStatus {
  available: boolean;
  path: string;
  reason?: string;
  representative?: boolean;
  epoch?: number | null;
  oil_iou?: number | null;
  size_mb?: number;
}

export interface SARAnalysis {
  image: { width: number; height: number };
  oil_fraction: number;
  coverage_percent: number;
  threshold: number;
  polygons_px: [number, number][][];
  slick_count: number;
  severity: { score: number; band: string; colour: string };
  model: {
    checkpoint: string;
    representative: boolean;
    epoch: number | null;
    oil_iou: number | null;
    caveat: string | null;
  };
  location: { lat: number; lon: number } | null;
}
