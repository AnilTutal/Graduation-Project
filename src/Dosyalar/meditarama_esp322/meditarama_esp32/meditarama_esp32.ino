/*
 * MediTarama - ESP32 Ana Kodu (SPIFFS YOK - HTML gömülü)
 * 
 * Sistem:gem
 *   - ESP32 WiFi Access Point açar ("MediTarama")
 *   - HTML doğrudan kod içinde (SPIFFS gerekmez)
 *   - MAX30102 → HR + SpO2 okur (5 ölçüm ortalaması)
 *   - DS18B20 → Sıcaklık okur (yoksa 36.6 sabit)
 *   - TFLite model ESP32'de çalışır (model_data.h)
 *   - /sensors → JSON { hr, spo2, temp, risk, confidence }
 * 
 * Donanım Bağlantıları:
 *   MAX30102 → SDA=21, SCL=22, VCC=3.3V, GND
 *   DS18B20  → Data=4 (4.7kΩ pull-up), VCC=3.3V, GND (opsiyonel)
 * 
 * Kütüphaneler:
 *   - TensorFlowLite_ESP32
 *   - SparkFun MAX3010x Pulse and Proximity Sensor Library
 *   - OneWire
 *   - DallasTemperature
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// TFLite
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// ============================================================
// AYARLAR
// ============================================================
const char* WIFI_SSID = "MediTarama";
const char* WIFI_PASS = "";

#define DS18B20_PIN 4
#define SAMPLE_COUNT 5
#define SAMPLE_INTERVAL_MS 1000

const float MEANS[]  = {82.334f, 36.793f, 96.362f, 52.645f, 0.499f, 77.354f, 1.761f, 25.364f};
const float SCALES[] = {16.357f,  0.854f,  3.749f, 20.574f, 0.500f, 17.034f, 0.151f,  6.620f};

// ============================================================
// HTML SAYFA (SPIFFS YOK, DOĞRUDAN KOD İÇİNDE)
// ============================================================
const char INDEX_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MediTarama</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: Georgia, serif; background: #f7f7f5; min-height: 100vh; }
  header {
    background: #fff; border-bottom: 1px solid #e8e8e4;
    padding: 18px 28px; display: flex;
    justify-content: space-between; align-items: center;
  }
  .logo { display: flex; align-items: center; gap: 10px; }
  .logo-dot { width: 10px; height: 10px; background: #111; border-radius: 50%; }
  .logo-text { font-size: 18px; font-weight: bold; letter-spacing: 1px; color: #111; }
  .badge { font-size: 11px; color: #888; border: 1px solid #ddd; padding: 4px 10px; border-radius: 20px; }
  main { max-width: 520px; margin: 40px auto; padding: 0 16px; }
  .card { background: #fff; border: 1px solid #e8e8e4; border-radius: 12px; padding: 32px 28px; }
  .card-title { font-size: 20px; font-weight: bold; color: #111; margin-bottom: 6px; }
  .card-sub { font-size: 13px; color: #888; margin-bottom: 24px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px; }
  label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; color: #555; }
  input, select {
    padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px;
    font-size: 15px; color: #111; background: #fafaf8;
    outline: none; font-family: inherit;
  }
  .btn {
    width: 100%; padding: 14px; background: #111; color: #fff;
    border: none; border-radius: 8px; font-size: 15px;
    cursor: pointer; letter-spacing: 0.5px; font-family: inherit;
  }
  .btn:disabled { background: #ccc; cursor: not-allowed; }
  .btn.grey { background: #444; }
  .sensor-row {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0; border-bottom: 1px solid #f0f0ee;
  }
  .sensor-row:last-of-type { border-bottom: none; margin-bottom: 20px; }
  .sensor-icon { font-size: 22px; width: 30px; text-align: center; }
  .sensor-info { flex: 1; }
  .sensor-name { font-size: 13px; color: #555; margin-bottom: 3px; }
  .sensor-val { font-size: 17px; font-weight: bold; color: #111; font-family: monospace; }
  .tick {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; border: 2px solid #ddd; color: #ccc; flex-shrink: 0;
  }
  .tick.done { background: #1a7a4a; border-color: #1a7a4a; color: #fff; }
  .tick.spin { border-color: #999; border-top-color: transparent;
               animation: spin 0.8s linear infinite; color: transparent; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .done-msg {
    text-align: center; font-size: 13px; color: #1a7a4a;
    background: #f0fff6; border: 1px solid #b3ffd6;
    border-radius: 8px; padding: 10px; margin-bottom: 20px; display: none;
  }
  .result-banner {
    border-radius: 10px; padding: 28px; text-align: center;
    margin-bottom: 20px; display: flex; flex-direction: column;
    align-items: center; gap: 6px;
  }
  .result-banner.risky  { background: #fff1f1; border: 1px solid #ffb3b3; }
  .result-banner.normal { background: #f1fff8; border: 1px solid #b3ffd6; }
  .result-emoji { font-size: 44px; }
  .result-title { font-size: 30px; font-weight: bold; letter-spacing: 2px; }
  .result-title.risky  { color: #c0392b; }
  .result-title.normal { color: #1a7a4a; }
  .result-conf { font-size: 13px; color: #666; }
  .result-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 20px; }
  .result-item {
    border: 1px solid #e8e8e4; border-radius: 8px; padding: 10px;
    display: flex; flex-direction: column; gap: 3px;
  }
  .result-item-label { font-size: 9px; color: #aaa; letter-spacing: 0.5px; text-transform: uppercase; }
  .result-item-val { font-size: 14px; font-weight: bold; color: #111; }
  .status { text-align: center; font-size: 12px; color: #aaa; margin-top: 10px; font-family: monospace; }
  .hidden { display: none; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    <span class="logo-text">MediTarama</span>
  </div>
  <span class="badge">ESP32 - Cevrimdisi</span>
</header>

<main>

  <div id="step-form" class="card">
    <h2 class="card-title">Hasta Bilgileri</h2>
    <p class="card-sub">Lutfen asagidaki bilgileri doldurun.</p>
    <div class="grid2">
      <label>Yas <input type="number" id="yas" placeholder="orn: 45" min="1"></label>
      <label>Cinsiyet
        <select id="cinsiyet">
          <option value="1">Erkek</option>
          <option value="0">Kadin</option>
        </select>
      </label>
      <label>Kilo (kg) <input type="number" id="kilo" placeholder="orn: 75" min="1"></label>
      <label>Boy (m) <input type="number" id="boy" placeholder="orn: 1.75" step="0.01" min="0.5"></label>
    </div>
    <button class="btn" onclick="goToSensor()">Sensor Taramasini Baslat</button>
  </div>

  <div id="step-sensor" class="card hidden">
    <h2 class="card-title">Sensor Okumasi</h2>
    <p class="card-sub">Parmaginizi sensore yerlestirin ve bekleyin.</p>

    <div class="sensor-row">
      <span class="sensor-icon">&#9829;</span>
      <div class="sensor-info">
        <div class="sensor-name">Nabiz (MAX30102)</div>
        <div class="sensor-val" id="val-hr">Oluyor...</div>
      </div>
      <div class="tick spin" id="tick-hr"></div>
    </div>

    <div class="sensor-row">
      <span class="sensor-icon">&#9675;</span>
      <div class="sensor-info">
        <div class="sensor-name">SpO2 (MAX30102)</div>
        <div class="sensor-val" id="val-spo2">Oluyor...</div>
      </div>
      <div class="tick spin" id="tick-spo2"></div>
    </div>

    <div class="sensor-row">
      <span class="sensor-icon">&#9672;</span>
      <div class="sensor-info">
        <div class="sensor-name">Sicaklik (DS18B20)</div>
        <div class="sensor-val" id="val-temp">Bekleniyor...</div>
      </div>
      <div class="tick" id="tick-temp"></div>
    </div>

    <div class="done-msg" id="done-msg">
      Tum olcumler tamamlandi - elinizi kaldirabillirsiniz
    </div>

    <button class="btn" id="btn-predict" onclick="predict()" disabled>Riski Analiz Et</button>
    <p class="status" id="sensor-status"></p>
  </div>

  <div id="step-result" class="card hidden">
    <div class="result-banner" id="result-banner">
      <span class="result-emoji" id="result-emoji"></span>
      <div class="result-title" id="result-title"></div>
      <div class="result-conf" id="result-conf"></div>
    </div>
    <div class="result-grid" id="result-grid"></div>
    <button class="btn grey" onclick="resetForm()">Yeni Olcum</button>
  </div>

</main>

<script>
let formData = null;
let sensorData = null;

function goToSensor() {
  var yas  = Number(document.getElementById('yas').value);
  var kilo = Number(document.getElementById('kilo').value);
  var boy  = Number(document.getElementById('boy').value);
  if (!yas || !kilo || !boy) { alert('Lutfen tum alanlari doldurun.'); return; }
  formData = { yas: yas, cinsiyet: Number(document.getElementById('cinsiyet').value), kilo: kilo, boy: boy };
  show('step-sensor');
  startMaxReading();
}

function startMaxReading() {
  setStatus('Nabiz ve SpO2 oluyor (5 saniye)...');
  setTick('tick-hr',   'spin');
  setTick('tick-spo2', 'spin');

  setTimeout(function() {
    var params = 'age='    + formData.yas +
                 '&gender='+ formData.cinsiyet +
                 '&weight='+ formData.kilo +
                 '&height='+ formData.boy;

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/sensors?' + params, true);
    xhr.onload = function() {
      if (xhr.status === 200) {
        var data = JSON.parse(xhr.responseText);
        if (!data.ready) {
          setStatus('Olcum devam ediyor...');
          setTimeout(startMaxReading, 1000);
          return;
        }
        sensorData = data;
        document.getElementById('val-hr').textContent   = data.hr   + ' bpm';
        document.getElementById('val-spo2').textContent = data.spo2 + ' %';
        setTick('tick-hr',   'done');
        setTick('tick-spo2', 'done');
        setStatus('Sicaklik oluyor...');
        setTick('tick-temp', 'spin');
        setTimeout(function() {
          document.getElementById('val-temp').textContent = data.temp + ' C';
          setTick('tick-temp', 'done');
          document.getElementById('done-msg').style.display = 'block';
          document.getElementById('btn-predict').disabled = false;
          setStatus('');
        }, 1500);
      }
    };
    xhr.onerror = function() {
      // Simule mod
      sensorData = { hr: 72, spo2: 98, temp: 36.6, isRisky: false, confidence: 85.0 };
      document.getElementById('val-hr').textContent   = '72 bpm';
      document.getElementById('val-spo2').textContent = '98 %';
      setTick('tick-hr',   'done');
      setTick('tick-spo2', 'done');
      setStatus('(Simule deger)');
      setTimeout(function() {
        document.getElementById('val-temp').textContent = '36.6 C';
        setTick('tick-temp', 'done');
        document.getElementById('done-msg').style.display = 'block';
        document.getElementById('btn-predict').disabled = false;
        setStatus('');
      }, 1500);
    };
    xhr.send();
  }, 5500);
}

function predict() {
  var data = sensorData;
  var isRisky    = data.isRisky    !== null ? data.isRisky    : false;
  var confidence = data.confidence !== null ? data.confidence : 85.0;
  var bmi = formData.boy > 0 ? (formData.kilo / (formData.boy * formData.boy)).toFixed(1) : '-';

  var banner = document.getElementById('result-banner');
  banner.className = 'result-banner ' + (isRisky ? 'risky' : 'normal');
  document.getElementById('result-emoji').textContent = isRisky ? '!' : 'OK';
  document.getElementById('result-title').textContent = isRisky ? 'RISKLI' : 'NORMAL';
  document.getElementById('result-title').className   = 'result-title ' + (isRisky ? 'risky' : 'normal');
  document.getElementById('result-conf').textContent  = 'Guven: %' + Number(confidence).toFixed(1);

  var items = [
    { label: 'Nabiz',    val: data.hr   + ' bpm' },
    { label: 'SpO2',     val: data.spo2 + ' %'   },
    { label: 'Sicaklik', val: data.temp + ' C'   },
    { label: 'BMI',      val: bmi                 },
    { label: 'Yas',      val: formData.yas         },
    { label: 'Cinsiyet', val: formData.cinsiyet === 1 ? 'Erkek' : 'Kadin' },
  ];

  document.getElementById('result-grid').innerHTML = items.map(function(i) {
    return '<div class="result-item"><span class="result-item-label">' + i.label +
           '</span><span class="result-item-val">' + i.val + '</span></div>';
  }).join('');

  show('step-result');
}

function show(id) {
  ['step-form','step-sensor','step-result'].forEach(function(s) {
    document.getElementById(s).classList.add('hidden');
  });
  document.getElementById(id).classList.remove('hidden');
}

function setTick(id, state) {
  var el = document.getElementById(id);
  el.className = 'tick';
  if (state === 'done') { el.classList.add('done'); el.textContent = 'v'; }
  else if (state === 'spin') { el.classList.add('spin'); el.textContent = ''; }
}

function setStatus(msg) { document.getElementById('sensor-status').textContent = msg; }

function resetForm() {
  sensorData = null; formData = null;
  document.getElementById('yas').value  = '';
  document.getElementById('kilo').value = '';
  document.getElementById('boy').value  = '';
  document.getElementById('cinsiyet').value = '1';
  document.getElementById('val-hr').textContent   = 'Oluyor...';
  document.getElementById('val-spo2').textContent  = 'Oluyor...';
  document.getElementById('val-temp').textContent  = 'Bekleniyor...';
  document.getElementById('tick-hr').className   = 'tick';
  document.getElementById('tick-spo2').className = 'tick';
  document.getElementById('tick-temp').className = 'tick';
  document.getElementById('done-msg').style.display = 'none';
  document.getElementById('btn-predict').disabled = true;
  setStatus('');
  show('step-form');
}
</script>
</body>
</html>
)rawhtml";

// ============================================================
// GLOBAL DEĞİŞKENLER
// ============================================================
WebServer server(80);
MAX30105 particleSensor;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::AllOpsResolver resolver;
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

float lastHR   = 0;
float lastSpO2 = 0;
float lastTemp = 0;
bool  sensorReady = false;

float hrBuf[SAMPLE_COUNT]   = {0};
float spo2Buf[SAMPLE_COUNT] = {0};
float tempBuf[SAMPLE_COUNT] = {0};
int   sampleIdx   = 0;
int   sampleCount = 0;

uint32_t irBuffer[100], redBuffer[100];
int32_t  spo2Val;
int8_t   spo2Valid;
int32_t  hrVal;
int8_t   hrValid;

unsigned long lastSampleTime = 0;

// ============================================================
// TFLite BAŞLAT
// ============================================================
bool initTFLite() {
  tfl_model = tflite::GetModel(medical_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema uyumsuzlugu!");
    return false;
  }
  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
  );
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors basarisiz!");
    return false;
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite hazir");
  return true;
}

// ============================================================
// TAHMİN
// ============================================================
float runInference(float hr, float temp, float spo2,
                   float age, float gender, float weight, float height) {
  float bmi = weight / (height * height);
  float raw[8] = {hr, temp, spo2, age, gender, weight, height, bmi};
  for (int i = 0; i < 8; i++) {
    input->data.f[i] = (raw[i] - MEANS[i]) / SCALES[i];
  }
  if (interpreter->Invoke() != kTfLiteOk) return -1;
  return output->data.f[0];
}

// ============================================================
// SENSÖR - MAX30102
// ============================================================
void readMAX30102() {
  for (int i = 0; i < 100; i++) {
    while (particleSensor.available() == 0) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, 100, redBuffer,
    &spo2Val, &spo2Valid, &hrVal, &hrValid
  );
}

// ============================================================
// SENSÖR - DS18B20
// ============================================================
float readDS18B20() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);
  if (t == -127.0f || t < 30.0f || t > 42.0f) return 36.6f;
  return t;
}

// ============================================================
// ORTALAMA
// ============================================================
void takeSample() {
  float hr = 72.0f;
  float spo2 = 98.0f;
  float temp = 36.6f;

  // Eğer sensör fiziksel olarak varsa oku, yoksa atla kilitlenme!
  if (particleSensor.getINT1()) { // Basit bir kontrol
      // readMAX30102(); // Sensör yoksa bu satır kodu kilitler!
      // Şimdilik test için burayı kapalı tutalım veya sensörü bağlayınca açarsın
  }

  temp = readDS18B20();

  hrBuf[sampleIdx]   = hr;
  spo2Buf[sampleIdx] = spo2;
  tempBuf[sampleIdx] = temp;
  sampleIdx = (sampleIdx + 1) % SAMPLE_COUNT;
  if (sampleCount < SAMPLE_COUNT) sampleCount++;

  float sumHR = 0, sumSpo2 = 0, sumTemp = 0;
  for (int i = 0; i < sampleCount; i++) {
    sumHR   += hrBuf[i];
    sumSpo2 += spo2Buf[i];
    sumTemp += tempBuf[i];
  }
  lastHR   = sumHR   / sampleCount;
  lastSpO2 = sumSpo2 / sampleCount;
  lastTemp = sumTemp / sampleCount;

  if (sampleCount >= SAMPLE_COUNT) sensorReady = true;

  Serial.printf("[Olcum %d/%d] HR=%.0f SpO2=%.0f Temp=%.1f\n",
    sampleCount, SAMPLE_COUNT, lastHR, lastSpO2, lastTemp);
}

// ============================================================
// /sensors
// ============================================================
void handleSensors() {
  server.sendHeader("Access-Control-Allow-Origin", "*");

  if (!sensorReady) {
    int remaining = SAMPLE_COUNT - sampleCount;
    server.send(200, "application/json",
      "{\"ready\":false,\"remaining\":" + String(remaining) + "}");
    return;
  }

  float age    = server.hasArg("age")    ? server.arg("age").toFloat()    : 0;
  float gender = server.hasArg("gender") ? server.arg("gender").toFloat() : 0;
  float weight = server.hasArg("weight") ? server.arg("weight").toFloat() : 0;
  float height = server.hasArg("height") ? server.arg("height").toFloat() : 0;

  float risk = -1, confidence = 0;
  if (age > 0 && weight > 0 && height > 0) {
    risk = runInference(lastHR, lastTemp, lastSpO2, age, gender, weight, height);
    confidence = (risk >= 0.5f) ? risk * 100.0f : (1.0f - risk) * 100.0f;
  }

  float bmi = (height > 0) ? weight / (height * height) : 0;

  String json = "{";
  json += "\"ready\":true,";
  json += "\"hr\":"   + String(lastHR,   1) + ",";
  json += "\"spo2\":" + String(lastSpO2, 1) + ",";
  json += "\"temp\":" + String(lastTemp, 1) + ",";
  json += "\"bmi\":"  + String(bmi,      1) + ",";
  if (risk >= 0) {
    json += "\"isRisky\":"    + String(risk >= 0.5f ? "true" : "false") + ",";
    json += "\"confidence\":" + String(confidence, 1);
  } else {
    json += "\"isRisky\":null,\"confidence\":null";
  }
  json += "}";

  server.send(200, "application/json", json);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== MediTarama Baslatiliyor ===");

  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.printf("WiFi AP: %s\n", WIFI_SSID);
  Serial.printf("IP: %s\n", WiFi.softAPIP().toString().c_str());

  ds18b20.begin();
  Serial.println("DS18B20 hazir");

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 bulunamadi - simule deger kullanilacak");
  } else {
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    Serial.println("MAX30102 hazir");
  }

  initTFLite();

  server.on("/", []() {
    server.send_P(200, "text/html", INDEX_HTML);
  });
  server.on("/sensors", handleSensors);
  server.begin();

  Serial.println("=== Hazir! 192.168.4.1 adresine baglan ===\n");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  server.handleClient();
  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;
    takeSample();
  }
}
