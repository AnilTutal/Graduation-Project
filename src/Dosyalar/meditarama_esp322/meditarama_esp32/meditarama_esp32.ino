#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <OneWire.h>
#include <DallasTemperature.h>
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <LittleFS.h>
// ─── EKRAN
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ─── WiFi
const char* WIFI_SSID = "MediTarama";
const char* WIFI_PASS = "";

// ─── PIN & SABITLER
#define DS18B20_PIN        4
#define SAMPLE_INTERVAL_MS 40   // her 500ms'de bir örnek
#define SPO2_BUFFER_LENGTH 100    // 50 × 500ms = 25 saniye

// ─── Normalizasyon
const float MEANS[]  = {82.334f, 36.793f, 96.362f, 52.645f, 0.499f, 77.354f, 1.761f, 25.364f};
const float SCALES[] = {16.357f,  0.854f,  3.749f, 20.574f, 0.500f, 17.034f, 0.151f,  6.620f};

// ─── NESNELER
WebServer server(80);
MAX30105 particleSensor;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

// ─── DURUM MAKİNESİ
// ADIM_1: HR+SpO2 ölçülüyor (parmak sensörde)
// ADIM_2: Sıcaklık ölçülüyor (parmak çekildi)
// HAZIR:  Her şey tamam, model çalışmaya hazır
enum Durum { BEKLIYOR, ADIM_1_HR_SPO2, ADIM_2_SICAKLIK, HAZIR };
Durum durum = BEKLIYOR;

// ─── Ölçüm sonuçları
float sonucHR    = 0;
float sonucSpO2  = 0;
float sonucTemp  = 0;
bool  maxFound   = false;

// ─── SpO2/HR buffer
uint32_t irBuffer[SPO2_BUFFER_LENGTH];
uint32_t redBuffer[SPO2_BUFFER_LENGTH];
int      bufferIndex = 0;

// ─── Sıcaklık örnekleme
#define TEMP_SAMPLE_COUNT  9   // Yaklaşık 15 saniyelik (9 x ~1.75sn) bir okuma periyodu için 9 örnek
float    tempSamples[TEMP_SAMPLE_COUNT];
int      tempIndex = 0;

unsigned long lastSampleTime = 0;

// ─── TFLite
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::AllOpsResolver     resolver;
  const tflite::Model*       tfl_model   = nullptr;
  tflite::MicroInterpreter*  interpreter = nullptr;
  TfLiteTensor*              inputTensor  = nullptr;
  TfLiteTensor*              outputTensor = nullptr;
  constexpr int kTensorArenaSize = 48 * 1024;  // 32→48KB artırıldı
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// ═══════════════════════════════════════════════════════════════
// HTML
// ═══════════════════════════════════════════════════════════════
const char INDEX_HTML[] PROGMEM = R"rawhtml(
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MediTarama</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
<style>
:root{--ink:#0e0e0e;--ink2:#3a3a3a;--ink3:#888;--paper:#faf9f6;--white:#fff;--rule:#e4e2dc;--rule2:#f0efe9;--teal:#0d7c6b;--teal-lt:#e8f5f2;--teal-mid:#b3ddd7;--rose:#c0392b;--rose-lt:#fdf0ee;--rose-mid:#f5c4be}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.6;background:var(--paper);color:var(--ink);min-height:100vh;display:flex;flex-direction:column;background-image:radial-gradient(ellipse 90% 40% at 50% 0%,rgba(13,124,107,.07) 0%,transparent 65%)}
header{height:58px;padding:0 28px;display:flex;align-items:center;justify-content:space-between;background:rgba(250,249,246,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--rule);position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:9px}
.logo-mark{width:28px;height:28px;background:var(--ink);border-radius:7px;display:grid;place-items:center;flex-shrink:0;color:white;font-family:'DM Serif Display',serif;font-size:16px;line-height:1}
.logo-text{font-family:'DM Serif Display',Georgia,serif;font-size:17px;color:var(--ink);letter-spacing:-.2px}
.sys-badge{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--teal);background:var(--teal-lt);border:1px solid var(--teal-mid);padding:3px 10px;border-radius:100px}
main{flex:1;display:flex;align-items:center;justify-content:center;padding:44px 16px}
.card{background:var(--white);border:1px solid var(--rule);border-radius:16px;padding:34px 30px;width:100%;max-width:480px;box-shadow:0 2px 8px rgba(0,0,0,.05),0 8px 32px rgba(0,0,0,.06);animation:rise .32s cubic-bezier(.2,0,0,1) both}
.card.wide{max-width:860px}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.card-label{font-size:10px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:var(--ink3);margin-bottom:5px}
.card h2{font-family:'DM Serif Display',Georgia,serif;font-size:25px;font-weight:400;color:var(--ink);letter-spacing:-.4px;margin-bottom:5px;line-height:1.2}
.card p.sub{font-size:13px;color:var(--ink3);margin-bottom:26px}
.select-grid{display:grid;gap:12px}
.select-card{display:flex;align-items:center;gap:16px;padding:18px 20px;border:1px solid var(--rule);border-radius:12px;cursor:pointer;background:var(--white);transition:border-color .18s,box-shadow .18s,transform .18s}
.select-card:hover{border-color:var(--ink);box-shadow:0 4px 16px rgba(0,0,0,.08);transform:translateY(-1px)}
.select-card:active{transform:translateY(0)}
.sel-icon{width:44px;height:44px;background:var(--rule2);border-radius:10px;display:grid;place-items:center;font-size:22px;flex-shrink:0}
.sel-body h4{font-size:15px;font-weight:600;color:var(--ink);margin-bottom:2px}
.sel-body p{font-size:12px;color:var(--ink3)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.igroup{margin-bottom:14px}
label{display:flex;flex-direction:column;gap:5px;font-size:11px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;color:var(--ink3)}
input,select{padding:10px 13px;border:1px solid var(--rule);border-radius:8px;background:var(--paper);font-family:'DM Sans',sans-serif;font-size:14px;color:var(--ink);width:100%;transition:border-color .15s,box-shadow .15s;appearance:none;-webkit-appearance:none}
input:focus,select:focus{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(13,124,107,.1);background:var(--white)}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23888' d='M5 6L0 0h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 13px center;padding-right:34px}
.btn{width:100%;padding:13px 20px;background:var(--ink);color:var(--white);border:none;border-radius:9px;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;cursor:pointer;letter-spacing:.1px;transition:background .15s,transform .1s;margin-top:4px}
.btn:hover{background:var(--ink2)}
.btn:active{transform:scale(.99)}
.btn:disabled{background:#bbb;cursor:not-allowed;transform:none}
.btn-ghost{width:100%;padding:11px;background:none;color:var(--ink3);border:none;border-radius:8px;font-family:'DM Sans',sans-serif;font-size:13px;cursor:pointer;margin-top:8px;transition:color .15s,background .15s}
.btn-ghost:hover{color:var(--ink);background:var(--rule2)}
.step-box{border:1px solid var(--rule);border-radius:11px;padding:15px 16px;margin-bottom:10px;background:var(--paper);transition:border-color .2s,background .2s,box-shadow .2s}
.step-box.active{border-color:var(--ink);background:var(--white);box-shadow:0 2px 10px rgba(0,0,0,.06)}
.step-box.done{border-color:var(--teal-mid);background:var(--teal-lt)}
.step-num{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--ink3);margin-bottom:3px}
.step-box.done .step-num{color:var(--teal)}
.step-title{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:2px}
.step-box.done .step-title{color:var(--teal)}
.step-sub{font-size:12px;color:var(--ink3)}
.progress-bg{height:4px;background:var(--rule);border-radius:4px;margin-top:10px;overflow:hidden}
.step-box.done .progress-bg{background:var(--teal-mid)}
.progress-fill{height:4px;background:var(--ink);border-radius:4px;width:0%;transition:width .5s cubic-bezier(.4,0,.2,1)}
.step-box.done .progress-fill{background:var(--teal)}
.sensor-rows{margin:18px 0 4px}
.sensor-row{display:flex;align-items:center;justify-content:space-between;padding:13px 0;border-bottom:1px solid var(--rule2)}
.sensor-row:last-child{border-bottom:none}
.lw{display:flex;flex-direction:column;gap:1px}
.lw .tech{font-size:9px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--ink3);font-style:normal}
.lw span{font-size:14px;color:var(--ink2);font-weight:400}
.sensor-row strong{font-size:18px;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.result-banner{border-radius:12px;padding:30px 24px;text-align:center;margin-bottom:20px}
.result-banner.normal{background:var(--teal-lt);color:var(--teal);border:1px solid var(--teal-mid)}
.result-banner.risky{background:var(--rose-lt);color:var(--rose);border:1px solid var(--rose-mid)}
.result-title{font-family:'DM Serif Display',Georgia,serif;font-size:34px;font-weight:400;margin-bottom:6px;letter-spacing:-.5px}
.result-conf{font-size:12.5px;opacity:.75;font-weight:500}
.metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}
.metric-box{background:var(--paper);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.m-label{font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--ink3);margin-bottom:5px}
.m-val{font-size:20px;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.table-wrap{width:100%;overflow-x:auto;border:1px solid var(--rule);border-radius:10px;margin-bottom:20px}
table{width:100%;border-collapse:collapse;min-width:700px}
thead{background:var(--paper)}
th{padding:11px 14px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--ink3);border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:12px 14px;border-bottom:1px solid var(--rule2);font-size:13px;color:var(--ink2)}
tr:last-child td{border-bottom:none}
td strong{color:var(--ink);font-weight:600}
.pill{display:inline-block;padding:3px 9px;border-radius:100px;font-size:10px;font-weight:700;letter-spacing:.4px;text-transform:uppercase}
.pill-n{background:var(--teal-lt);color:var(--teal);border:1px solid var(--teal-mid)}
.pill-r{background:var(--rose-lt);color:var(--rose);border:1px solid var(--rose-mid)}
.btn-del{color:var(--rose);cursor:pointer;border:none;background:none;font-size:16px;padding:2px 6px;border-radius:5px;transition:background .15s}
.btn-del:hover{background:var(--rose-lt)}
.empty-row td{text-align:center;padding:36px;color:var(--ink3);font-size:13px}
.patient-tag{display:inline-flex;align-items:center;gap:6px;background:var(--rule2);border:1px solid var(--rule);border-radius:100px;padding:4px 12px;font-size:12px;font-weight:500;color:var(--ink2);margin-bottom:18px}
.hidden{display:none!important}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-text">MediTarama</div>
  </div>
</header>

<main>

  <!-- EKRAN 0: Giris -->
  <div id="step-login" class="card">
    <div class="card-label">Giriş</div>
    <h2>Hoş Geldiniz</h2>
    <p class="sub">İşlem yapmak istediğiniz profili seçin</p>
    <div class="select-grid">
      <div class="select-card" onclick="show('step-form')">
        <div class="sel-icon">👤</div>
        <div class="sel-body">
          <h4>Yeni Analiz</h4>
          <p>Hasta verilerini girin ve analiz başlatın</p>
        </div>
      </div>
      <div class="select-card" onclick="openAdmin()">
        <div class="sel-icon">⚕️</div>
        <div class="sel-body">
          <h4>Doktor Paneli</h4>
          <p>Geçmiş analiz kayıtlarını görüntüleyin</p>
        </div>
      </div>
    </div>
  </div>

  <!-- EKRAN 1: Hasta Bilgileri -->
  <div id="step-form" class="card hidden">
    <div class="card-label">Adım 1 / 3</div>
    <h2>Hasta Bilgileri</h2>
    <p class="sub">Analiz için hasta bilgilerini doldurun</p>
    <div class="igroup">
      <label>Ad Soyad <input type="text" id="adSoyad" placeholder="Örn: Ahmet Yılmaz"></label>
    </div>
    <div class="grid2">
      <div class="igroup">
        <label>Yaş <input type="number" id="yas" value="45" min="1" max="120"></label>
      </div>
      <div class="igroup">
        <label>Cinsiyet
          <select id="cinsiyet">
            <option value="1">Erkek</option>
            <option value="0">Kadın</option>
          </select>
        </label>
      </div>
      <div class="igroup">
        <label>Kilo (kg) <input type="number" id="kilo" value="75" min="20" max="300"></label>
      </div>
      <div class="igroup">
        <label>Boy (m) <input type="number" id="boy" value="1.75" min="0.5" max="2.5" step="0.01"></label>
      </div>
    </div>
    <button class="btn" onclick="goToSensor()">Analizi Başlat</button>
    <button class="btn-ghost" onclick="show('step-login')">Vazgeç</button>
  </div>

  <!-- EKRAN 2: Sensor -->
  <div id="step-sensor" class="card hidden">
    <div class="card-label">Adım 2 / 3</div>
    <h2>Ölçüm Aşamaları</h2>
    <div class="patient-tag" id="sensor-patient-name"></div>

    <div class="step-box active" id="box-hr">
      <div class="step-num">Adım 1 / 2</div>
      <div class="step-title" id="hr-title">Parmağınızı sensöre yerleştirin</div>
      <div class="step-sub"   id="hr-sub">Nabız ve oksijen ölçülüyor</div>
      <div class="progress-bg"><div class="progress-fill" id="prog-hr"></div></div>
    </div>

    <div class="step-box" id="box-temp">
      <div class="step-num">Adım 2 / 2</div>
      <div class="step-title" id="temp-title">Parmağınızı çekin</div>
      <div class="step-sub"   id="temp-sub">Sıcaklık ölçümü başlayacak</div>
      <div class="progress-bg"><div class="progress-fill" id="prog-temp"></div></div>
    </div>

    <div class="sensor-rows">
      <div class="sensor-row">
        <div class="lw"><em class="tech">MAX30102</em><span>Nabız</span></div>
        <strong id="val-hr">--</strong>
      </div>
      <div class="sensor-row">
        <div class="lw"><em class="tech">MAX30102</em><span>SpO2</span></div>
        <strong id="val-spo2">--</strong>
      </div>
      <div class="sensor-row">
        <div class="lw"><em class="tech">DS18B20</em><span>Sıcaklık</span></div>
        <strong id="val-temp">--</strong>
      </div>
    </div>

    <button class="btn" id="btn-predict" onclick="predict()" disabled style="margin-top:10px">
      AI Analizini Gör
    </button>
    <button class="btn-ghost" id="btn-reset" onclick="resetOlcum()" style="margin-top:6px">
      Ölçümü Yeniden Başlat
    </button>
  </div>

  <!-- EKRAN 3: Sonuc -->
  <div id="step-result" class="card hidden">
    <div class="card-label">Adım 3 / 3</div>
    <h2>Analiz Sonucu</h2>
    <div id="result-banner" class="result-banner" style="margin-top:14px">
      <div class="result-title" id="result-status">---</div>
      <div class="result-conf"  id="result-conf">---</div>
    </div>
    <div class="metrics-grid" id="metrics-grid"></div>
    <button class="btn" onclick="location.reload()">Ana Menüye Dön</button>
  </div>

  <!-- EKRAN 4: Admin -->
  <div id="step-admin" class="card wide hidden">
    <div class="card-label">Doktor Paneli</div>
    <h2>Analiz Kayıtları</h2>
    <p class="sub" style="margin-bottom:18px">ESP32 hafızasındaki son 10 kayıt</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ad Soyad</th><th>Yaş</th><th>BMI</th>
            <th>Nabız</th><th>SpO2</th><th>Sıcaklık</th>
            <th>Durum</th><th>Güven</th><th>Sil</th>
          </tr>
        </thead>
        <tbody id="admin-body"></tbody>
      </table>
    </div>
    <button class="btn" style="max-width:180px" onclick="show('step-login')">Geri Dön</button>
  </div>

</main>

<script>
let formData   = null;
let sensorData = null;
let lastState  = '';
let historyData = [];

function show(id) {
  document.querySelectorAll('.card').forEach(c => c.classList.add('hidden'));
  document.getElementById(id).classList.remove('hidden');
  if (id === 'step-login') {
    fetch('/reset', { method: 'POST' }).catch(() => {});
    formData = null; sensorData = null; lastState = '';
  }
}

function goToSensor() {
  const ad = document.getElementById('adSoyad').value.trim();
  if (!ad) { alert('Lütfen Ad Soyad giriniz.'); return; }
  formData = {
    ad: ad,
    yas: document.getElementById('yas').value,
    g:   document.getElementById('cinsiyet').value,
    w:   document.getElementById('kilo').value,
    h:   document.getElementById('boy').value
  };
  document.getElementById('sensor-patient-name').innerText = ad;
  show('step-sensor');
  startPolling();
}

function startPolling() {
  let poller = setInterval(() => {
    fetch('/sensors?age=' + formData.yas + '&gender=' + formData.g +
          '&weight=' + formData.w + '&height=' + formData.h +
          '&name=' + encodeURIComponent(formData.ad), { cache: 'no-store' })
      .then(r => r.json())
      .then(d => {
        if (d.hr   > 0) document.getElementById('val-hr').innerText   = d.hr   + ' bpm';
        if (d.spo2 > 0) document.getElementById('val-spo2').innerText = d.spo2 + ' %';
        if (d.temp > 0) document.getElementById('val-temp').innerText = d.temp + ' °C';
        const pct = Math.round(d.pct || 0);
        if (d.state === 'hr') {
          document.getElementById('prog-hr').style.width = pct + '%';
          document.getElementById('hr-title').innerText  = 'Nabiz & Oksijen olcuuluyor... (%' + pct + ')';
          if (lastState !== 'hr') {
            document.getElementById('box-hr').className   = 'step-box active';
            document.getElementById('box-temp').className = 'step-box';
          }
        }
        if (d.state === 'temp') {
          document.getElementById('prog-hr').style.width   = '100%';
          document.getElementById('hr-title').innerText    = '✓ Nabiz & Oksijen alindi';
          document.getElementById('box-hr').className      = 'step-box done';
          document.getElementById('box-temp').className    = 'step-box active';
          document.getElementById('prog-temp').style.width = pct + '%';
          document.getElementById('temp-title').innerText  = 'Sicaklik olculuyor... (%' + pct + ')';
          document.getElementById('temp-sub').innerText    = 'DS18B20 oluyor — sabit bekleyin';
        }
        if (d.state === 'ready') {
          clearInterval(poller);
          sensorData = d;
          document.getElementById('prog-hr').style.width   = '100%';
          document.getElementById('prog-temp').style.width = '100%';
          document.getElementById('hr-title').innerText    = '✓ Nabiz & Oksijen alindi';
          document.getElementById('temp-title').innerText  = '✓ Sicaklik alindi';
          document.getElementById('box-hr').className      = 'step-box done';
          document.getElementById('box-temp').className    = 'step-box done';
          document.getElementById('btn-predict').disabled  = false;
        }
        lastState = d.state;
      })
      .catch(() => {
        document.getElementById('hr-sub').innerText = 'ESP32 baglantisi bekleniyor...';
      });
  }, 1000);
}

function predict() {
  const isRisky = sensorData.isRisky === 'true';
  const bmi     = (parseFloat(formData.w) / (parseFloat(formData.h) * parseFloat(formData.h))).toFixed(1);
  const banner  = document.getElementById('result-banner');
  banner.className = 'result-banner ' + (isRisky ? 'risky' : 'normal');
  document.getElementById('result-status').innerText = isRisky ? '⚠ RISKLI' : '✓ NORMAL';
  document.getElementById('result-conf').innerText   = 'Yapay Zeka Guven Orani: %' + sensorData.confidence;
  const metrics = [
    { label: 'Nabiz',    val: sensorData.hr   + ' bpm' },
    { label: 'SpO2',     val: sensorData.spo2 + ' %'   },
    { label: 'Sicaklik', val: sensorData.temp + ' °C'  },
    { label: 'BMI',      val: bmi }
  ];
  document.getElementById('metrics-grid').innerHTML = metrics.map(m =>
    '<div class="metric-box"><div class="m-label">' + m.label + '</div><div class="m-val">' + m.val + '</div></div>'
  ).join('');
  const record = {
    ad: formData.ad, yas: formData.yas, bmi,
    hr: sensorData.hr, spo2: sensorData.spo2, temp: sensorData.temp,
    isRisky, confidence: sensorData.confidence
  };
  fetch('/save', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(record) }).catch(()=>{});
  show('step-result');
}

function openAdmin() {
  fetch('/history', { cache: 'no-store' })
    .then(r => r.json()).then(data => { historyData = data; renderAdmin(); })
    .catch(() => renderAdmin());
}

function renderAdmin() {
  const body = document.getElementById('admin-body');
  if (!historyData || historyData.length === 0) {
    body.innerHTML = '<tr class="empty-row"><td colspan="9">Henuz kayit bulunmuyor.</td></tr>';
  } else {
    body.innerHTML = historyData.map((h, i) =>
      '<tr>' +
      '<td><strong>' + h.ad + '</strong></td>' +
      '<td>' + h.yas + '</td><td>' + h.bmi + '</td>' +
      '<td>' + h.hr + ' bpm</td><td>' + h.spo2 + '%</td><td>' + h.temp + '°C</td>' +
      '<td><span class="pill ' + (h.isRisky ? 'pill-r' : 'pill-n') + '">' + (h.isRisky ? 'RISKLI' : 'NORMAL') + '</span></td>' +
      '<td>%' + h.confidence + '</td>' +
      '<td><button class="btn-del" onclick="deleteItem(' + i + ')">×</button></td>' +
      '</tr>'
    ).join('');
  }
  show('step-admin');
}

function deleteItem(index) {
  if (!confirm('Bu kaydi silmek istediginize emin misiniz?')) return;
  fetch('/delete?index=' + index, { method: 'POST' })
    .then(() => openAdmin())
    .catch(() => { historyData.splice(index, 1); renderAdmin(); });
}

 function resetOlcum() {
      fetch('/reset', { method: 'POST' })
        .then(() => {
          lastState = '';
          sensorData = null;
          document.getElementById('prog-hr').style.width   = '0%';
          document.getElementById('prog-temp').style.width = '0%';
          document.getElementById('hr-title').innerText    = 'Parmaginizi sensore yerlestirin';
          document.getElementById('hr-sub').innerText      = 'Nabiz ve oksijen olculuyor - 5 saniye sabit tutun';
          document.getElementById('temp-title').innerText  = 'Parmaginizi cekin';
          document.getElementById('temp-sub').innerText    = 'Sicaklik olcumu baslayacak - 5 saniye';
          document.getElementById('box-hr').className      = 'step-box active';
          document.getElementById('box-temp').className    = 'step-box';
          document.getElementById('val-hr').innerText      = '--';
          document.getElementById('val-spo2').innerText    = '--';
          document.getElementById('val-temp').innerText    = '--';
          document.getElementById('btn-predict').disabled  = true;
          startPolling();
        })
        .catch(() => {});
    }
</script>
</body>
</html>
)rawhtml";

struct Record {
  char  ad[32];
  int   yas;
  float bmi, hr, spo2, temp, confidence;
  bool  isRisky;
};
#define MAX_RECORDS 10
Record records[MAX_RECORDS];
int    recordCount = 0;

// ═══════════════════════════════════════════════════════════════
// TFLite
// ═══════════════════════════════════════════════════════════════
bool initTFLite() {
  tfl_model = tflite::GetModel(medical_model);

  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[AI] ✘ Schema uyumsuz! Model=%d Beklenen=%d\n",
                  tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[AI] ✘ AllocateTensors başarısız! Arena boyutunu artır.");
    return false;
  }

  inputTensor  = interpreter->input(0);
  outputTensor = interpreter->output(0);

  Serial.println("[AI] ✔ TFLite hazır.");
  Serial.printf("[AI]   Input  shape: [%d, %d]\n",
                inputTensor->dims->data[0], inputTensor->dims->data[1]);
  Serial.printf("[AI]   Output shape: [%d, %d]\n",
                outputTensor->dims->data[0], outputTensor->dims->data[1]);
  return true;
}

float runInference(float hr, float temp, float spo2,
                   float age, float gender, float weight, float height) {
  float bmi    = weight / (height * height);
  float raw[8] = {hr, temp, spo2, age, gender, weight, height, bmi};

  Serial.println("[AI] ── Inference ──────────────────────────");
  const char* names[8] = {"HR","Temp","SpO2","Age","Gender","Weight","Height","BMI"};
  for (int i = 0; i < 8; i++) {
    float n = (raw[i] - MEANS[i]) / SCALES[i];
    inputTensor->data.f[i] = n;
    Serial.printf("[AI]   %-7s ham=%.2f  norm=%.4f\n", names[i], raw[i], n);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[AI] ✘ Invoke başarısız!");
    return -1.0f;
  }

  float r = outputTensor->data.f[0];
  Serial.printf("[AI]   Ham çıktı: %.6f  →  %s\n", r, r >= 0.5f ? "RİSKLİ" : "NORMAL");
  Serial.println("[AI] ───────────────────────────────────────");
  return r;
}

// ═══════════════════════════════════════════════════════════════
// DS18B20
// ═══════════════════════════════════════════════════════════════
float readDS18B20() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);
  if (t < 20.0f || t > 45.0f) return 36.6f;
  return t;
}

// ═══════════════════════════════════════════════════════════════
// OLED
// ═══════════════════════════════════════════════════════════════
void updateOLED() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  switch (durum) {
    case BEKLIYOR:
      display.setCursor(4, 20); display.print("Parmagi sensore");
      display.setCursor(4, 36); display.print("yerlestirin");
      break;

    case ADIM_1_HR_SPO2: {
      int pct = (bufferIndex * 100) / SPO2_BUFFER_LENGTH;
      display.setCursor(0, 0);  display.print("ADIM 1: HR & SpO2");
      display.setCursor(0, 16); display.print("HR:   "); display.print(sonucHR,   0); display.print(" bpm");
      display.setCursor(0, 30); display.print("SpO2: "); display.print(sonucSpO2, 0); display.print(" %");
      display.setCursor(0, 48); display.print("Ilerleme: %"); display.print(pct);
      break;
    }

    case ADIM_2_SICAKLIK: {
      int pct = (tempIndex * 100) / TEMP_SAMPLE_COUNT;
      display.setCursor(0, 0);  display.print("ADIM 2: Sicaklik");
      display.setCursor(0, 16); display.print("HR:   "); display.print(sonucHR,   0); display.print(" bpm");
      display.setCursor(0, 30); display.print("SpO2: "); display.print(sonucSpO2, 0); display.print(" %");
      display.setCursor(0, 48); display.print("Ilerleme: %"); display.print(pct);
      break;
    }

    case HAZIR:
      display.setCursor(0, 0);  display.print("HR:   "); display.print(sonucHR,   0); display.print(" bpm");
      display.setCursor(0, 18); display.print("SpO2: "); display.print(sonucSpO2, 0); display.print(" %");
      display.setCursor(0, 36); display.print("Isi:  "); display.print(sonucTemp, 1); display.print(" C");
      display.setCursor(0, 54); display.print("[ HAZIR - Analiz Et ]");
      break;
  }

  display.display();
}

// ═══════════════════════════════════════════════════════════════
// ÖRNEK ALMA — her 500ms çağrılır
// ═══════════════════════════════════════════════════════════════
void takeSample() {
  particleSensor.check();

  if (durum == BEKLIYOR) {
    if (particleSensor.available()) {
      long ir = particleSensor.getFIFOIR();
      long red = particleSensor.getFIFORed();
      particleSensor.nextSample();
      
      if (ir > 30000 && maxFound) {
        durum = ADIM_1_HR_SPO2;
        bufferIndex = 0;
        Serial.println("\n[ADIM 1] Başladı -> Parmağınızı sabit tutun...");
      }
    }
    return;
  }

  if (durum == ADIM_1_HR_SPO2) {
    while (particleSensor.available()) {
      long red = particleSensor.getFIFORed();
      long ir = particleSensor.getFIFOIR();
      particleSensor.nextSample();

      if (ir < 20000) { 
        Serial.println("[ADIM 1] ⚠ Parmak çekildi, BEKLEME moduna dönülüyor.");
        durum = BEKLIYOR;
        bufferIndex = 0;
        return;
      }

      // Buffer'a veri ekle
      redBuffer[bufferIndex] = (uint32_t)red;
      irBuffer[bufferIndex]  = (uint32_t)ir;
      bufferIndex++;

      // Her 25 örnekte bir ilerleme logu bas
      if (bufferIndex % 25 == 0 && bufferIndex < SPO2_BUFFER_LENGTH) {
        Serial.printf("[ADIM 1] Veri toplanıyor: %% %d | Ham IR: %ld\n", 
                      (bufferIndex * 100 / SPO2_BUFFER_LENGTH), ir);
      }

      if (bufferIndex >= SPO2_BUFFER_LENGTH) {
        int32_t spo2Raw; int8_t spo2Valid;
        int32_t hrRaw;   int8_t hrValid;

        Serial.println("[ADIM 1] Analiz penceresi doldu, Maxim algoritması çalıştırılıyor...");

        maxim_heart_rate_and_oxygen_saturation(
            irBuffer, SPO2_BUFFER_LENGTH, redBuffer,
            &spo2Raw, &spo2Valid, &hrRaw, &hrValid);

        // --- LOG EKLEMESİ ---
        Serial.println("─── Ölçüm Sonuçları ───");
        Serial.printf("> Nabız Geçerli mi? : %s\n", hrValid ? "EVET" : "HAYIR");
        Serial.printf("> Ham Nabız Değeri   : %d bpm\n", hrRaw);
        Serial.printf("> SpO2 Geçerli mi?  : %s\n", spo2Valid ? "EVET" : "HAYIR");
        Serial.printf("> Ham SpO2 Değeri   : %d %%\n", spo2Raw);
        Serial.println("──────────────────────");

        if (hrValid && spo2Valid && hrRaw > 40 && hrRaw < 180 && spo2Raw > 80) {
          sonucHR = (float)hrRaw;
          sonucSpO2 = (float)spo2Raw;
          
          durum = ADIM_2_SICAKLIK;
          tempIndex = 0;
          Serial.printf("[ADIM 1] ✔ BAŞARILI. Sıcaklık ölçümüne geçiliyor.\n");
        } else {
          Serial.println("[ADIM 1] ⚠ Geçersiz veya kararsız veri tespit edildi. Kaydırmalı pencereye geçiliyor...");
          // Kaydırmalı pencere (Sliding Window): Son 25 veriyi tutarak baştan başlamak yerine kesintisiz analiz yapar
          for (int i = 25; i < SPO2_BUFFER_LENGTH; i++) {
            redBuffer[i - 25] = redBuffer[i];
            irBuffer[i - 25] = irBuffer[i];
          }
          bufferIndex = SPO2_BUFFER_LENGTH - 25; 
        }
        break; // FIFO'yu bir sonraki iterasyonda kontrol et
      }
    }
    return;
  }

  if (durum == ADIM_2_SICAKLIK) {
    // DS18B20 okuması yavaş olduğu için her saniye 1 kere okuyacağız (1 saniye = 25 * 40ms)
    static int subCounter = 0;
    subCounter++;

    if (subCounter >= 25) { 
      subCounter = 0; // Sayacı sıfırla
      
      float t = readDS18B20();
      tempSamples[tempIndex++] = t;

      Serial.printf("[ADIM 2] %d/25 Saniye | Sıcaklık: %.2f°C | İlerleme: %%%d\n", 
                    tempIndex, t, (tempIndex * 100 / TEMP_SAMPLE_COUNT));

      if (tempIndex % 5 == 0) {
        Serial.println(">> Ölçüm devam ediyor, lütfen bekleyin...");
      }
    }

    if (tempIndex >= TEMP_SAMPLE_COUNT) {
      float sum = 0;
      for (int i = 0; i < TEMP_SAMPLE_COUNT; i++) sum += tempSamples[i];
      sonucTemp = sum / TEMP_SAMPLE_COUNT;
      durum = HAZIR;
      Serial.printf("[ADIM 2] ✔ TAMAMLANDI. Ortalama Sıcaklık: %.2f°C\n", sonucTemp);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// WEB HANDLER
// ═══════════════════════════════════════════════════════════════
void handleSensors() {
  float a = server.arg("age").toFloat();
  float g = server.arg("gender").toFloat();
  float w = server.arg("weight").toFloat();
  float h = server.arg("height").toFloat();

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-cache, no-store");

  String state;
  int    pct = 0;
  String json;

  switch (durum) {
    case BEKLIYOR:
      state = "hr"; pct = 0;
      json = "{\"state\":\"hr\",\"pct\":0"
             ",\"hr\":0,\"spo2\":0,\"temp\":0"
             ",\"isRisky\":\"false\",\"confidence\":0}";
      Serial.println("[WEB] Durum: BEKLIYOR");
      break;

    case ADIM_1_HR_SPO2:
      pct  = (bufferIndex * 100) / SPO2_BUFFER_LENGTH;
      json = "{\"state\":\"hr\",\"pct\":" + String(pct) +
             ",\"hr\":"   + String(sonucHR,   1) +
             ",\"spo2\":" + String(sonucSpO2, 1) +
             ",\"temp\":" + String(sonucTemp, 1) +
             ",\"isRisky\":\"false\",\"confidence\":0}";
      //Serial.printf("[WEB] Durum: ADIM_1_HR_SPO2  %%  %d\n", pct);
      break;

    case ADIM_2_SICAKLIK:
      pct  = (tempIndex * 100) / TEMP_SAMPLE_COUNT;
      json = "{\"state\":\"temp\",\"pct\":" + String(pct) +
             ",\"hr\":"   + String(sonucHR,   1) +
             ",\"spo2\":" + String(sonucSpO2, 1) +
             ",\"temp\":" + String(sonucTemp, 1) +
             ",\"isRisky\":\"false\",\"confidence\":0}";
      //Serial.printf("[WEB] Durum: ADIM_2_SICAKLIK  %%  %d\n", pct);
      break;

    case HAZIR: {
      float r = runInference(sonucHR, sonucTemp, sonucSpO2, a, g, w, h);

      if (isnan(r) || isinf(r) || r < 0.0f) {
        Serial.printf("[AI] ✘ Geçersiz çıktı: %.6f\n", r);
        json = "{\"state\":\"ready\""
               ",\"hr\":"   + String(sonucHR,   1) +
               ",\"spo2\":" + String(sonucSpO2, 1) +
               ",\"temp\":" + String(sonucTemp, 1) +
               ",\"isRisky\":\"false\",\"confidence\":0"
               ",\"error\":\"Model hatası\"}";
      } else {
        bool  risky = (r >= 0.5f);
        float conf  = risky ? r * 100.0f : (1.0f - r) * 100.0f;
        json = "{\"state\":\"ready\""
               ",\"hr\":"         + String(sonucHR,   1) +
               ",\"spo2\":"       + String(sonucSpO2, 1) +
               ",\"temp\":"       + String(sonucTemp, 1) +
               ",\"isRisky\":\""  + String(risky ? "true" : "false") + "\""
               ",\"confidence\":" + String(conf, 1) + "}";
        Serial.printf("[WEB] Sonuç: %s  Güven: %.1f%%\n",
                      risky ? "RİSKLİ" : "NORMAL", conf);
      }
      break;
    }
  }

  server.send(200, "application/json", json);
}

// ═══════════════════════════════════════════════════════════════
// SAVE (NEW)
// ═══════════════════════════════════════════════════════════════
void saveToInternalFlash() {
  File file = LittleFS.open("/history.dat", "w");
  if (!file) return;
  file.write((uint8_t*)&recordCount, sizeof(int));
  file.write((uint8_t*)records, sizeof(Record) * recordCount);
  file.close();
  Serial.println("[Hafiza] Veriler ESP32 icine kalici olarak yazildi.");
}

// ═══════════════════════════════════════════════════════════════
// SAVE
// ═══════════════════════════════════════════════════════════════
void handleSave() {
  if (!server.hasArg("plain")) { server.send(400); return; }
  String body = server.arg("plain");
  auto extract = [&](const char* key) -> String {
    String k = String("\"") + key + "\":";
    int idx = body.indexOf(k);
    if (idx < 0) return "";
    idx += k.length();
    if (body[idx] == '"') { int end = body.indexOf('"', idx+1); return body.substring(idx+1, end); }
    int end = body.indexOf(',', idx); if (end < 0) end = body.indexOf('}', idx);
    return body.substring(idx, end);
  };
  if (recordCount < MAX_RECORDS) recordCount++;
  for (int i = recordCount-1; i > 0; i--) records[i] = records[i-1];
  strncpy(records[0].ad, extract("ad").c_str(), 31);
  records[0].yas        = extract("yas").toInt();
  records[0].bmi        = extract("bmi").toFloat();
  records[0].hr         = extract("hr").toFloat();
  records[0].spo2       = extract("spo2").toFloat();
  records[0].temp       = extract("temp").toFloat();
  records[0].isRisky    = (extract("isRisky") == "true");
  records[0].confidence = extract("confidence").toFloat();
  Serial.printf("[KAYIT] %s kaydedildi. Toplam: %d\n", records[0].ad, recordCount);
  saveToInternalFlash();  // ← bu satırı ekle
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", "{\"ok\":true}");
}

void handleHistory() {
  String json = "[";
  for (int i = 0; i < recordCount; i++) {
    if (i > 0) json += ",";
    json += "{\"ad\":\"" + String(records[i].ad) + "\","
            "\"yas\":"   + String(records[i].yas)        + ","
            "\"bmi\":"   + String(records[i].bmi, 1)     + ","
            "\"hr\":"    + String(records[i].hr,  1)     + ","
            "\"spo2\":"  + String(records[i].spo2, 1)    + ","
            "\"temp\":"  + String(records[i].temp, 1)    + ","
            "\"isRisky\":" + String(records[i].isRisky ? "true" : "false") + ","
            "\"confidence\":" + String(records[i].confidence, 1) + "}";
  }
  json += "]";
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void handleDelete() {
  if (!server.hasArg("index")) { server.send(400); return; }
  int idx = server.arg("index").toInt();
  if (idx < 0 || idx >= recordCount) { server.send(400); return; }
  for (int i = idx; i < recordCount-1; i++) records[i] = records[i+1];
  recordCount--;
  Serial.printf("[DELETE] %d. kayit silindi. Kalan: %d\n", idx, recordCount);
  saveToInternalFlash();
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", "{\"ok\":true}");
}

// ═══════════════════════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════════════════════
void handleReset() {
  durum       = BEKLIYOR;
  bufferIndex = 0;
  tempIndex   = 0;
  sonucHR     = 0;
  sonucSpO2   = 0;
  sonucTemp   = 0;
  Serial.println("[RESET] Olcum sifirlandi, yeni olcum bekleniyor.");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", "{\"ok\":true}");
}

// ═══════════════════════════════════════════════════════════════
// SETUP & LOOP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n════════════════════════════════════════════");
  Serial.println("         MediTarama Baslatiliyor            ");
  Serial.println("════════════════════════════════════════════");

  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi]     ✔ IP: "); Serial.println(WiFi.softAPIP());

  ds18b20.begin();
  Serial.println("[DS18B20]  ✔ Baslatildi.");

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("[OLED]     ✘ Ekran bulunamadi!");
  } else {
    display.clearDisplay();
    display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
    display.setCursor(20, 24); display.print("MediTarama");
    display.display();
    Serial.println("[OLED]     ✔ Hazir.");
  }

  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    // 60 LED akımı, 4 örnek ortalama, 2 LED modu (Red+IR), 100Hz örnekleme hızı
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    maxFound = true;
    Serial.println("[MAX30105] ✔ Hazir.");
  } else {
    Serial.println("[MAX30105] ✘ Bulunamadi!");
  }

  initTFLite();

  server.on("/", []() { server.send_P(200, "text/html", INDEX_HTML); });
  server.on("/sensors", handleSensors);
  server.on("/save",    HTTP_POST, handleSave);
  server.on("/history", HTTP_GET,  handleHistory);
  server.on("/delete",  HTTP_POST, handleDelete);
  server.on("/reset", HTTP_POST, handleReset);
  if(!LittleFS.begin(true)){
    Serial.println("[Hafiza] LittleFS hatasi!");
  } else {
    File file = LittleFS.open("/history.dat", "r");
    if(file){
      file.read((uint8_t*)&recordCount, sizeof(int));
      file.read((uint8_t*)records, sizeof(Record) * recordCount);
      file.close();
      Serial.printf("[Hafiza] %d kayit ESP32 hafizasindan geri yuklendi!\n", recordCount);
    }
  }
  server.begin();
  Serial.println("[WEB]      ✔ HTTP sunucu baslatildi.");
  Serial.println("════════════════════════════════════════════\n");
}

void loop() {
  server.handleClient();
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = millis();
    takeSample();
    updateOLED();
  }
}

