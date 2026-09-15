import type { Analysis, Drift, Mission } from "./types";

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export interface DeclaredPose {
  lat: number;
  lon: number;
  altitude_m: number;
  pitch_deg: number;
  yaw_deg: number;
  hfov_deg: number;
}

export async function analyse(
  file: File,
  pose: DeclaredPose,
  ecologicalSensitivity: number,
  // Floating litter is small and far from the camera, so genuine detections
  // land well below the threshold tuned for close-up COCO objects. 0.12 is
  // where a bottle on open water becomes visible without the frame filling
  // with noise.
  sensitivity = 0.12,
): Promise<Analysis> {
  const form = new FormData();
  form.append("image", file);
  form.append("pose", JSON.stringify(pose));
  form.append("ecological_sensitivity", String(ecologicalSensitivity));
  form.append("sensitivity", String(sensitivity));
  return json<Analysis>(await fetch("/api/analyse", { method: "POST", body: form }));
}

export async function drift(
  lat: number,
  lon: number,
  classId: string,
  hours: number,
  reverse = false,
): Promise<Drift> {
  return json<Drift>(
    await fetch("/api/drift", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, class_id: classId, hours, reverse }),
    }),
  );
}

export async function planMission(params: {
  session_id: string;
  base_lat: number;
  base_lon: number;
  drift_aware: boolean;
  cruise_speed_ms: number;
  payload_capacity_kg: number;
  endurance_min: number;
}): Promise<Mission> {
  return json<Mission>(
    await fetch("/api/mission/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function resolveGate(
  missionId: string,
  index: number,
  decision: "approve" | "skip" | "flag",
): Promise<Mission> {
  return json<Mission>(
    await fetch(`/api/mission/${missionId}/gate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index, decision }),
    }),
  );
}

export function missionSocket(missionId: string): WebSocket {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${protocol}://${location.host}/ws/mission/${missionId}`);
}

export async function sarStatus(): Promise<import("./types").SARStatus> {
  return json(await fetch("/api/sar/status"));
}

export async function analyseSAR(
  file: File,
  threshold: number,
  pose: DeclaredPose,
): Promise<import("./types").SARAnalysis> {
  const form = new FormData();
  form.append("image", file);
  form.append("threshold", String(threshold));
  form.append("pose", JSON.stringify(pose));
  return json(await fetch("/api/sar/analyse", { method: "POST", body: form }));
}
