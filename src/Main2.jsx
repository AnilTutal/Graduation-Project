import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

const MEANS  = [82.334, 36.793, 96.362, 52.645, 0.499, 77.354, 1.761, 25.364];
const SCALES = [16.357,  0.854,  3.749, 20.574, 0.500, 17.034, 0.151,  6.620];

function pulse(color) {
  return `0 0 0 0 ${color}`;
}

// ESP32 gelince sensör simülasyonu kısmını fetch('http://192.168.4.1/sensors') ile değiştirmen yeterli — geri her şey aynı kalır.

export default function Main2() {
  const [model, setModel]       = useState(null);
  const [loading, setLoading]   = useState(true);
  const [step, setStep]         = useState("form"); // form | sensor | result
  const [form, setForm]         = useState({ yas: "", cinsiyet: "1", kilo: "", boy: "" });
  const [sensor, setSensor]     = useState({ hr: null, spo2: null, temp: null });
  const [result, setResult]     = useState(null);
  const [scanning, setScanning] = useState(false);
  const [dots, setDots]         = useState("");
  const timerRef = useRef(null);

  // Model yükle
  useEffect(() => {
    tf.loadLayersModel("/model/model.json")
      .then(m => { setModel(m); setLoading(false); })
      .catch(() => { setLoading(false); });
  }, []);

  // Nokta animasyonu
  useEffect(() => {
    if (scanning) {
      timerRef.current = setInterval(() => {
        setDots(d => d.length >= 3 ? "" : d + ".");
      }, 400);
    } else {
      clearInterval(timerRef.current);
      setDots("");
    }
    return () => clearInterval(timerRef.current);
  }, [scanning]);

  const handleForm = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const formValid = form.yas && form.kilo && form.boy &&
    Number(form.yas) > 0 && Number(form.kilo) > 0 && Number(form.boy) > 0;

  // Sensör simülasyonu (ESP32 gelince burası fetch olacak)
  const scanSensors = () => {
    setScanning(true);
    setStep("sensor");
    setTimeout(() => {
      setSensor({
        hr:   Math.round(60 + Math.random() * 40),
        spo2: Math.round(94 + Math.random() * 5),
        temp: (36.2 + Math.random() * 1.5).toFixed(1),
      });
      setScanning(false);
    }, 3000);
  };

  const predict = () => {
    if (!model || !sensor.hr) return;
    const w   = Number(form.kilo);
    const h   = Number(form.boy);
    const bmi = w / (h * h);
    const raw = [
      sensor.hr, Number(sensor.temp), sensor.spo2,
      Number(form.yas), Number(form.cinsiyet), w, h, bmi
    ];
    const scaled = raw.map((v, i) => (v - MEANS[i]) / SCALES[i]);
    const inp    = tf.tensor2d([scaled]);
    const prob   = model.predict(inp).dataSync()[0];
    setResult({ prob, bmi, isRisky: prob >= 0.5 });
    setStep("result");
  };

  const reset = () => {
    setStep("form");
    setResult(null);
    setSensor({ hr: null, spo2: null, temp: null });
  };

  if (loading) return (
    <div style={styles.center}>
      <div style={styles.spinner} />
      <p style={{ color: "#888", marginTop: 16, fontFamily: "monospace" }}>Model yükleniyor...</p>
    </div>
  );

  return (
    <div style={styles.root}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoDot} />
          <span style={styles.logoText}>MediTarama</span>
        </div>
        <span style={styles.badge}>ESP32 · Çevrimdışı</span>
      </header>

      <main style={styles.main}>

        {/* ADIM 1: FORM */}
        {step === "form" && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Hasta Bilgileri</h2>
            <p style={styles.cardSub}>Lütfen aşağıdaki bilgileri doldurun.</p>

            <div style={styles.grid2}>
              <label style={styles.label}>
                Yaş
                <input name="yas" type="number" value={form.yas}
                  onChange={handleForm} style={styles.input} placeholder="örn: 45" />
              </label>
              <label style={styles.label}>
                Cinsiyet
                <select name="cinsiyet" value={form.cinsiyet}
                  onChange={handleForm} style={styles.input}>
                  <option value="1">Erkek</option>
                  <option value="0">Kadın</option>
                </select>
              </label>
              <label style={styles.label}>
                Kilo (kg)
                <input name="kilo" type="number" value={form.kilo}
                  onChange={handleForm} style={styles.input} placeholder="örn: 75" />
              </label>
              <label style={styles.label}>
                Boy (m)
                <input name="boy" type="number" step="0.01" value={form.boy}
                  onChange={handleForm} style={styles.input} placeholder="örn: 1.75" />
              </label>
            </div>

            <button
              onClick={scanSensors}
              disabled={!formValid}
              style={{ ...styles.btn, ...(formValid ? {} : styles.btnDisabled) }}
            >
              Sensör Taramasını Başlat →
            </button>
          </div>
        )}

        {/* ADIM 2: SENSÖR */}
        {step === "sensor" && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Sensör Okuması</h2>
            <p style={styles.cardSub}>MAX30102 ve DS18B20 sensörlerinden veri alınıyor.</p>

            <div style={styles.sensorGrid}>
              {[
                { label: "Nabız", unit: "bpm", val: sensor.hr, icon: "♥" },
                { label: "SpO2",  unit: "%",   val: sensor.spo2, icon: "○" },
                { label: "Sıcaklık", unit: "°C", val: sensor.temp, icon: "◈" },
              ].map(({ label, unit, val, icon }) => (
                <div key={label} style={styles.sensorBox}>
                  <span style={styles.sensorIcon}>{icon}</span>
                  <span style={styles.sensorLabel}>{label}</span>
                  <span style={styles.sensorVal}>
                    {scanning ? <span style={styles.scanDot}>{dots || "·"}</span> : `${val} ${unit}`}
                  </span>
                </div>
              ))}
            </div>

            {!scanning && sensor.hr && (
              <button onClick={predict} style={styles.btn}>
                Riski Analiz Et →
              </button>
            )}

            {scanning && (
              <p style={{ textAlign: "center", color: "#aaa", fontFamily: "monospace" }}>
                Okunuyor{dots}
              </p>
            )}
          </div>
        )}

        {/* ADIM 3: SONUÇ */}
        {step === "result" && result && (
          <div style={styles.card}>
            <div style={{
              ...styles.resultBanner,
              background: result.isRisky ? "#fff1f1" : "#f1fff8",
              borderColor: result.isRisky ? "#ffb3b3" : "#b3ffd6",
            }}>
              <span style={{ fontSize: 48 }}>{result.isRisky ? "⚠️" : "✅"}</span>
              <h2 style={{
                ...styles.resultTitle,
                color: result.isRisky ? "#c0392b" : "#1a7a4a"
              }}>
                {result.isRisky ? "RİSKLİ" : "NORMAL"}
              </h2>
              <p style={styles.resultConf}>
                Güven: %{((result.isRisky ? result.prob : 1 - result.prob) * 100).toFixed(1)}
              </p>
            </div>

            <div style={styles.resultGrid}>
              {[
                { label: "Nabız",     val: `${sensor.hr} bpm` },
                { label: "SpO2",      val: `${sensor.spo2} %` },
                { label: "Sıcaklık", val: `${sensor.temp} °C` },
                { label: "BMI",       val: result.bmi.toFixed(1) },
                { label: "Yaş",       val: form.yas },
                { label: "Cinsiyet",  val: form.cinsiyet === "1" ? "Erkek" : "Kadın" },
              ].map(({ label, val }) => (
                <div key={label} style={styles.resultItem}>
                  <span style={styles.resultItemLabel}>{label}</span>
                  <span style={styles.resultItemVal}>{val}</span>
                </div>
              ))}
            </div>

            <button onClick={reset} style={{ ...styles.btn, background: "#333" }}>
              ← Yeni Ölçüm
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#f7f7f5",
    fontFamily: "'Georgia', serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "20px 32px",
    borderBottom: "1px solid #e8e8e4",
    background: "#fff",
  },
  logo: { display: "flex", alignItems: "center", gap: 10 },
  logoDot: {
    width: 10, height: 10, borderRadius: "50%",
    background: "#111", display: "inline-block",
  },
  logoText: { fontSize: 18, fontWeight: "bold", letterSpacing: 1, color: "#111" },
  badge: {
    fontSize: 11, color: "#888", border: "1px solid #ddd",
    padding: "4px 10px", borderRadius: 20, letterSpacing: 0.5,
  },
  main: {
    maxWidth: 520,
    margin: "48px auto",
    padding: "0 16px",
  },
  card: {
    background: "#fff",
    border: "1px solid #e8e8e4",
    borderRadius: 12,
    padding: "36px 32px",
  },
  cardTitle: { fontSize: 22, fontWeight: "bold", color: "#111", margin: "0 0 6px" },
  cardSub: { fontSize: 14, color: "#888", margin: "0 0 28px" },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 28 },
  label: { display: "flex", flexDirection: "column", gap: 6, fontSize: 13, color: "#555" },
  input: {
    padding: "10px 12px",
    border: "1px solid #ddd",
    borderRadius: 8,
    fontSize: 15,
    color: "#111",
    outline: "none",
    background: "#fafaf8",
  },
  btn: {
    width: "100%",
    padding: "14px",
    background: "#111",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 15,
    cursor: "pointer",
    letterSpacing: 0.5,
  },
  btnDisabled: { background: "#ccc", cursor: "not-allowed" },
  sensorGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: 12,
    marginBottom: 28,
  },
  sensorBox: {
    border: "1px solid #e8e8e4",
    borderRadius: 10,
    padding: "16px 12px",
    textAlign: "center",
    background: "#fafaf8",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  sensorIcon: { fontSize: 20 },
  sensorLabel: { fontSize: 11, color: "#aaa", letterSpacing: 0.5 },
  sensorVal: { fontSize: 16, fontWeight: "bold", color: "#111", fontFamily: "monospace" },
  scanDot: { color: "#bbb" },
  resultBanner: {
    border: "1px solid",
    borderRadius: 10,
    padding: "28px",
    textAlign: "center",
    marginBottom: 24,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
  },
  resultTitle: { fontSize: 32, fontWeight: "bold", margin: 0, letterSpacing: 2 },
  resultConf: { fontSize: 14, color: "#666", margin: 0 },
  resultGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap: 10,
    marginBottom: 24,
  },
  resultItem: {
    border: "1px solid #e8e8e4",
    borderRadius: 8,
    padding: "10px 12px",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  resultItemLabel: { fontSize: 10, color: "#aaa", letterSpacing: 0.5 },
  resultItemVal: { fontSize: 14, fontWeight: "bold", color: "#111" },
  center: {
    display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    minHeight: "100vh",
  },
  spinner: {
    width: 32, height: 32,
    border: "3px solid #eee",
    borderTop: "3px solid #111",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
};