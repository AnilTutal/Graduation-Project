import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';


const Main = () => {
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  // Senin Scaler Sabitlerin (Python çıktından aldıklarımız)
  const means = [82.334, 36.793, 96.362, 52.645, 0.499, 77.354, 1.761, 25.364];
  const scales = [16.357, 0.854, 3.749, 20.574, 0.500, 17.034, 0.151, 6.620];
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
    const fullData = [...rawData, bmi];

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
 

  
  
{/**
const Main = () => {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [testResults, setTestResults] = useState(null);

  // Senin Scaler Sabitlerin (Python çıktından aldıklarımız)
  // NOT: Modelin 9 parametreli ise 9 elemanlı kalsın, 8 ise son elemanları silmelisin.
  const means = [82.334, 36.793, 96.362, 52.645, 0.499, 77.354, 1.761, 25.364];
const scales = [16.357, 0.854, 3.749, 20.574, 0.500, 17.034, 0.151, 6.620];
  // 100 Test Senaryosu ve Beklenen Sonuçlar
  const rawScenarios = [
    [72, 36.6, 98, 25, 1, 75, 1.80], [105, 36.8, 97, 30, 0, 60, 1.65], [65, 36.5, 92, 45, 1, 85, 1.75], [80, 38.2, 96, 12, 0, 40, 1.50], [70, 36.7, 96, 78, 1, 70, 1.70],
    [55, 36.4, 98, 50, 0, 65, 1.60], [85, 36.8, 99, 40, 1, 110, 1.75], [75, 37.1, 97, 60, 0, 55, 1.62], [125, 39.0, 88, 80, 1, 75, 1.72], [68, 36.6, 96, 76, 0, 60, 1.60],
    [115, 37.8, 93, 22, 1, 70, 1.80], [70, 36.6, 96, 76, 0, 65, 1.65], [88, 36.5, 95, 77, 1, 80, 1.75], [130, 36.7, 98, 28, 1, 90, 1.85], [65, 38.8, 97, 35, 0, 70, 1.60],
    [75, 36.8, 89, 50, 1, 85, 1.80], [95, 36.6, 96, 40, 0, 120, 1.60], [45, 36.2, 97, 60, 1, 75, 1.75], [82, 36.9, 98, 79, 0, 55, 1.62], [72, 36.5, 96, 45, 1, 50, 1.85],
    [101, 37.4, 96, 20, 1, 70, 1.75], [60, 36.0, 94, 33, 0, 58, 1.68], [74, 36.6, 98, 85, 1, 80, 1.80], [110, 39.5, 91, 5, 1, 18, 1.10], [90, 37.0, 97, 40, 0, 95, 1.60],
    [58, 36.2, 98, 22, 1, 72, 1.82], [120, 37.5, 90, 45, 0, 80, 1.65], [72, 36.6, 98, 30, 1, 70, 1.75], [76, 36.8, 97, 35, 0, 62, 1.68], [140, 37.2, 99, 19, 1, 75, 1.85],
    [66, 35.0, 95, 55, 0, 70, 1.65], [88, 36.9, 85, 65, 1, 90, 1.75], [70, 36.6, 98, 80, 0, 50, 1.55], [102, 38.6, 94, 25, 1, 85, 1.80], [50, 36.4, 96, 42, 0, 60, 1.70],
    [78, 37.4, 97, 77, 1, 88, 1.78], [92, 36.8, 98, 20, 0, 100, 1.55], [70, 36.6, 98, 40, 1, 70, 1.75], [110, 36.9, 97, 30, 0, 55, 1.60], [74, 39.2, 96, 50, 1, 80, 1.75],
    [65, 36.7, 93, 60, 0, 70, 1.65], [105, 37.8, 94, 15, 1, 55, 1.70], [72, 36.6, 98, 28, 0, 55, 1.65], [80, 36.8, 97, 82, 1, 75, 1.70], [118, 38.1, 92, 40, 0, 90, 1.60],
    [60, 36.5, 96, 35, 1, 130, 1.85], [95, 37.5, 95, 25, 0, 65, 1.70], [70, 36.6, 98, 10, 1, 30, 1.40], [130, 39.5, 85, 70, 0, 60, 1.55], [72, 36.6, 98, 35, 1, 78, 1.80],
    [70, 36.5, 98, 22, 1, 70, 1.75], [115, 39.2, 92, 28, 0, 55, 1.60], [60, 36.6, 99, 45, 1, 105, 1.80], [82, 37.1, 94, 65, 0, 75, 1.55], [145, 37.0, 97, 19, 1, 80, 1.85],
    [75, 36.4, 98, 5, 1, 18, 1.10],   [52, 35.8, 96, 70, 0, 60, 1.60],  [90, 38.5, 95, 35, 1, 85, 1.75],  [65, 36.7, 88, 50, 0, 70, 1.65],  [110, 36.8, 96, 82, 1, 65, 1.72],
    [72, 36.6, 98, 25, 0, 52, 1.68],  [102, 37.5, 93, 12, 1, 40, 1.50], [68, 36.5, 97, 77, 0, 58, 1.62], [125, 39.5, 85, 40, 1, 90, 1.80], [88, 36.9, 95, 30, 0, 120, 1.65],
    [62, 36.2, 98, 21, 1, 70, 1.85],  [95, 37.8, 91, 55, 0, 80, 1.58],  [74, 36.8, 97, 85, 1, 75, 1.70], [135, 37.2, 98, 27, 0, 65, 1.70], [58, 36.0, 94, 42, 1, 88, 1.75],
    [80, 36.6, 99, 10, 0, 30, 1.45],  [105, 38.8, 92, 3, 1, 14, 0.95],  [70, 36.7, 96, 68, 0, 72, 1.60], [118, 37.4, 90, 48, 1, 95, 1.78], [66, 36.4, 98, 33, 0, 50, 1.60],
    [78, 36.8, 97, 24, 1, 140, 1.80], [92, 37.6, 95, 15, 0, 45, 1.55],  [50, 36.1, 96, 58, 1, 82, 1.75], [122, 39.1, 87, 72, 0, 63, 1.55], [74, 36.6, 98, 40, 1, 78, 1.82],
    [108, 37.9, 94, 20, 0, 55, 1.65], [64, 36.5, 93, 80, 1, 68, 1.70],  [85, 37.0, 97, 38, 0, 110, 1.60], [130, 38.5, 96, 25, 1, 85, 1.85], [76, 36.7, 98, 29, 0, 60, 1.68],
    [55, 36.3, 91, 62, 1, 85, 1.75],  [98, 37.5, 94, 18, 0, 48, 1.62],  [72, 36.6, 99, 75, 1, 70, 1.75], [112, 39.4, 89, 45, 0, 75, 1.65], [68, 36.8, 97, 10, 1, 35, 1.40],
    [84, 37.2, 95, 55, 0, 90, 1.58],  [140, 37.8, 98, 30, 1, 80, 1.80], [60, 36.5, 92, 79, 0, 55, 1.60], [101, 38.6, 93, 22, 1, 72, 1.78], [74, 36.6, 98, 50, 0, 65, 1.65],
    [95, 37.3, 96, 35, 1, 115, 1.80], [48, 35.9, 97, 65, 0, 60, 1.55],  [120, 36.9, 91, 26, 1, 85, 1.85], [78, 39.0, 95, 12, 0, 42, 1.52], [70, 36.5, 98, 28, 1, 75, 1.78]
  ];

  const expectedLabels = [
    0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0
  ];

  useEffect(() => {
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

  const runBulkTest = async () => {
    let correctCount = 0;
    const results = [];

    for (let i = 0; i < rawScenarios.length; i++) {
      const s = rawScenarios[i];
      const bmi = s[5] / (s[6] ** 2);
      
      
      // FullData oluştur (Python ile aynı sıra: HR, Temp, SpO2, Age, Gen, W, H, BMI, HRV)
      const fullData = [...s, bmi];

      // Normalizasyon
      const scaledData = fullData.map((val, idx) => (val - means[idx]) / scales[idx]);

      // Tahmin
      const inputTensor = tf.tensor2d([scaledData]);
      const predictionResult = model.predict(inputTensor);
      const score = await predictionResult.data();
      
      const isRisky = score[0] > 0.5;
      const expectedIsRisky = expectedLabels[i] === 1;
      const isCorrect = isRisky === expectedIsRisky;

      if (isCorrect) correctCount++;

      results.push({
        id: i + 1,
        expected: expectedIsRisky ? "RİSKLİ" : "NORMAL",
        predicted: isRisky ? "RİSKLİ" : "NORMAL",
        confidence: (isRisky ? score[0] : 1 - score[0]) * 100,
        status: isCorrect ? "✅" : "❌"
      });
    }

    setTestResults({
      accuracy: (correctCount / rawScenarios.length) * 100,
      details: results
    });
  };

  if (loading) return <h2>🧠 Model Hazırlanıyor...</h2>;

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>📊 Model Otomatik Test Paneli</h1>
      <p>Bu panel, 100 farklı senaryoyu TensorFlow.js üzerinden test eder.</p>
      
      <button 
        onClick={runBulkTest} 
        style={{ padding: '15px 30px', fontSize: '18px', cursor: 'pointer', background: '#2ecc71', color: 'white', border: 'none', borderRadius: '5px' }}
      >
        🚀 100 Senaryo Testini Başlat
      </button>

      {testResults && (
        <div style={{ marginTop: '30px' }}>
          <div style={{ padding: '20px', background: '#f8f9fa', borderRadius: '10px', marginBottom: '20px' }}>
            <h2>🎯 Toplam Başarı Oranı: %{testResults.accuracy.toFixed(2)}</h2>
            <p>100 Senaryodan { (testResults.accuracy * 100) / 100 } tanesi doğru tahmin edildi.</p>
          </div>

          <table border="1" cellPadding="10" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center' }}>
            <thead>
              <tr style={{ background: '#34495e', color: 'white' }}>
                <th>No</th>
                <th>Beklenen</th>
                <th>Tahmin Edilen</th>
                <th>Güven Oranı</th>
                <th>Durum</th>
              </tr>
            </thead>
            <tbody>
              {testResults.details.map((res) => (
                <tr key={res.id} style={{ backgroundColor: res.status === "❌" ? "#ff000022" : "transparent" }}>
                  <td>{res.id}</td>
                  <td>{res.expected}</td>
                  <td>{res.predicted}</td>
                  <td>%{res.confidence.toFixed(2)}</td>
                  <td>{res.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Main;
     */}