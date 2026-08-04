import { useEffect, useState } from "react";
import { api, Forecast, Station } from "./api";

type Stage = "signup" | "verify" | "done";

export function App() {
  const [stations, setStations] = useState<Station[]>([]);
  const [station, setStation] = useState("");
  const [phone, setPhone] = useState("");
  const [mode, setMode] = useState("nightly");
  const [threshold, setThreshold] = useState(0);
  const [code, setCode] = useState("");
  const [stage, setStage] = useState<Stage>("signup");
  const [error, setError] = useState("");
  const [forecasts, setForecasts] = useState<Forecast[]>([]);

  useEffect(() => {
    api
      .stations()
      .then((s) => {
        setStations(s);
        if (s.length) setStation(s[0].key);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  async function submitSignup(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      await api.subscribe({ station, phone, mode, threshold_c: threshold });
      setStage("verify");
    } catch (e: any) {
      setError(e.message ?? "could not sign up");
    }
  }

  async function submitVerify(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      await api.verify({ phone, station, code });
      setForecasts(await api.forecasts(station));
      setStage("done");
    } catch (e: any) {
      setError(e.message ?? "verification failed");
    }
  }

  return (
    <main>
      <h1>❄️ Frost Detector</h1>
      <p className="lede">
        An evening frost forecast for growers. We text you tomorrow morning's
        predicted minimum temperature — a forecast, not a warning.
      </p>

      {error && <p className="error">{error}</p>}

      {stage === "signup" && (
        <form onSubmit={submitSignup}>
          <label>
            Station
            <select value={station} onChange={(e) => setStation(e.target.value)}>
              {stations.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Text me
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="nightly">every night</option>
              <option value="frost">only when frost is forecast</option>
            </select>
          </label>

          {mode === "frost" && (
            <label>
              Warn me below (°C)
              <input
                type="number"
                step="0.5"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
            </label>
          )}

          <label>
            Phone
            <input
              type="tel"
              placeholder="+44…"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              required
            />
          </label>

          <button type="submit">Sign up</button>
        </form>
      )}

      {stage === "verify" && (
        <form onSubmit={submitVerify}>
          <p>We sent a code to {phone}. Enter it to activate forecasts.</p>
          <label>
            Verification code
            <input
              value={code}
              onChange={(e) => setCode(e.target.value)}
              required
            />
          </label>
          <button type="submit">Verify</button>
        </form>
      )}

      {stage === "done" && (
        <section>
          <p className="ok">You're subscribed. Recent forecasts:</p>
          {forecasts.length === 0 ? (
            <p>No forecasts stored yet — the first arrives after tonight's run.</p>
          ) : (
            <ul className="forecasts">
              {forecasts.map((f) => (
                <li key={f.date}>
                  <strong>{f.date}</strong>: {f.predicted_tmin_c.toFixed(1)} °C
                  {f.alarm_fired ? " — frost likely" : ""}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
