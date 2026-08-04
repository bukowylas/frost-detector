// Typed client for the Frost Detector backend. Same-origin: the API is served by
// the same FastAPI app that serves this build, so paths are relative.

export interface Station {
  key: string;
  label: string;
}

export interface Forecast {
  station: string;
  date: string;
  predicted_tmin_c: number;
  alarm_fired: boolean;
  model_version: string;
  cutoff_ts: string;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  stations: () => fetch("/api/stations").then((r) => json<Station[]>(r)),

  subscribe: (body: {
    station: string;
    phone: string;
    mode: string;
    threshold_c: number;
  }) =>
    fetch("/api/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ id: number; station: string; verified: boolean }>(r)),

  verify: (body: { phone: string; station: string; code: string }) =>
    fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ verified: boolean }>(r)),

  forecasts: (station: string) =>
    fetch(`/api/forecasts/${station}`).then((r) => json<Forecast[]>(r)),
};
