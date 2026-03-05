import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const Main = () => {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  // Senin Scaler Sabitlerin (Python çıktından aldıklarımız)
  const means = [79.533, 36.743, 97.499, 53.433, 0.499, 74.988, 1.744, 25.155, 1.284];
  const scales = [11.553, 0.433, 1.442, 20.796, 0.499, 14.468, 0.144, 6.497, 0.191];

  useEffect(() => {
    // Modeli yükle
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        setModel(loadedModel);
        setLoading(false);
        console.log("Model başarıyla yüklendi!");
      } catch (err) {
        console.error("Model yükleme hatası:", err);
      }
    };
    loadModel();
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const rawData = [
      Number(formData.get('hr')),
      Number(formData.get('temp')),
      Number(formData.get('spo2')),
      Number(formData.get('age')),
      formData.get('gender') === 'Male' ? 1 : 0,
      Number(formData.get('weight')),
      Number(formData.get('height'))
    ];

    // BMI ve HRV hesapla (Senin Python kodundaki mantık)
    const bmi = rawData[5] / (rawData[6] ** 2);
    const hrv = 1 / (rawData[0] * 0.01);
    const fullData = [...rawData, bmi, hrv];

    // Normalizasyon (Z-Score)
    const scaledData = fullData.map((val, i) => (val - means[i]) / scales[i]);

    // Tahmin yap
    const inputTensor = tf.tensor2d([scaledData]);
    const result = model.predict(inputTensor);
    const score = await result.data();

    setPrediction({
      isRisky: score[0] > 0.5,
      confidence: (score[0] > 0.5 ? score[0] : 1 - score[0]) * 100,
      calculatedBMI: bmi, // Ekranda göstermek için state'e ekledik
      calculatedHRV: hrv  // Ekranda göstermek için state'e ekledik
    });
  };

  if (loading) return <h2>Yapay Zeka Yükleniyor...</h2>;

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>🏥 Risk Tahmin Paneli</h1>
      <form onSubmit={handlePredict} style={{ display: 'grid', gap: '10px', maxWidth: '400px' }}>
        <input name="hr" type="number" placeholder="Nabız (HR)" required />
        <input name="temp" type="number" step="0.1" placeholder="Ateş" required />
        <input name="spo2" type="number" placeholder="SpO2" required />
        <input name="age" type="number" placeholder="Yaş" required />
        <select name="gender"><option>Male</option><option>Female</option></select>
        <input name="weight" type="number" placeholder="Kilo (kg)" required />
        <input name="height" type="number" step="0.01" placeholder="Boy (m)" required />
        <button type="submit" style={{ padding: '10px', background: 'blue', color: 'white' }}>Tahmin Et</button>
      </form>

      {prediction && (
        <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: prediction.isRisky ? '#ffe6e6' : '#e6fffa' }}>
    <h3>Sonuç: {prediction.isRisky ? '🚨 RİSKLİ' : '✅ NORMAL'}</h3>
    <p>Güven Skoru: %{prediction.confidence.toFixed(2)}</p>
    <hr />
    <p><strong>Hesaplanan BMI:</strong> {prediction.calculatedBMI.toFixed(2)}</p>
    <p><strong>Türetilmiş HRV:</strong> {prediction.calculatedHRV.toFixed(2)}</p>
  </div>
      )}
    </div>
  );
};

export default Main;