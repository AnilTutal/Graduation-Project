/*
 * MediTarama - ESP32 Ana Kodu
 * 
 * Sistem:
 *   - ESP32 WiFi Access Point açar ("MediTarama")
 *   - SPIFFS'ten index.html serve eder
 *   - MAX30102 → HR + SpO2 okur (5 ölçüm ortalaması)
 *   - DS18B20 → Sıcaklık okur
 *   - TFLite model ESP32'de çalışır (model_data.h)
 *   - /sensors → JSON { hr, spo2, temp, risk, confidence }
 * 
 * Donanım Bağlantıları:
 *   MAX30102 → SDA=21, SCL=22, VCC=3.3V, GND
 *   DS18B20  → Data=4 (4.7kΩ pull-up gerekli), VCC=3.3V, GND
 * 
 * Kütüphaneler (Arduino Library Manager'dan kur):
 *   - TensorFlowLite_ESP32
 *   - SparkFun MAX3010x Pulse and Proximity Sensor Library
 *   - OneWire
 *   - DallasTemperature
 * 
 * SPIFFS'e yükle (Tools → ESP32 Sketch Data Upload):
 *   data/index.html
 */

#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
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
const char* WIFI_PASS = "";  // Şifresiz

#define DS18B20_PIN 4
#define SAMPLE_COUNT 5
#define SAMPLE_INTERVAL_MS 1000

// SCALER SABİTLERİ (Python eğitiminden)
const float MEANS[]  = {82.334f, 36.793f, 96.362f, 52.645f, 0.499f, 77.354f, 1.761f, 25.364f};
const float SCALES[] = {16.357f,  0.854f,  3.749f, 20.574f, 0.500f, 17.034f, 0.151f,  6.620f};

// ============================================================
// GLOBAL DEĞİŞKENLER
// ============================================================
WebServer server(80);
MAX30105 particleSensor;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

// TFLite
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

// Sensör değerleri
float lastHR   = 0;
float lastSpO2 = 0;
float lastTemp = 0;
bool  sensorReady = false;

// Ölçüm tamponları
float hrBuf[SAMPLE_COUNT]   = {0};
float spo2Buf[SAMPLE_COUNT] = {0};
float tempBuf[SAMPLE_COUNT] = {0};
int   sampleIdx = 0;
int   sampleCount = 0;

// MAX30102
uint32_t irBuffer[100], redBuffer[100];
int32_t  spo2Val;
int8_t   spo2Valid;
int32_t  hrVal;
int8_t   hrValid;

// Zamanlama
unsigned long lastSampleTime = 0;

// ============================================================
// TFLite BAŞLAT
// ============================================================
bool initTFLite() {
  tfl_model = tflite::GetModel(medical_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite schema uyumsuzluğu!");
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors başarısız!");
    return false;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("✅ TFLite hazır");
  return true;
}

// ============================================================
// TAHMIN YAP
// ============================================================
float runInference(float hr, float temp, float spo2,
                   float age, float gender, float weight, float height) {
  float bmi = weight / (height * height);
  float raw[8] = {hr, temp, spo2, age, gender, weight, height, bmi};

  for (int i = 0; i < 8; i++) {
    input->data.f[i] = (raw[i] - MEANS[i]) / SCALES[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke başarısız!");
    return -1;
  }

  return output->data.f[0];
}

// ============================================================
// SENSÖR OKUMA - MAX30102
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
// SENSÖR OKUMA - DS18B20
// ============================================================
float readDS18B20() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);
  if (t == -127.0f || t < 30.0f || t > 42.0f) return 36.6f;
  return t;
}

// ============================================================
// ORTALAMA AL VE SENSÖR HAZIR YAP
// ============================================================
void takeSample() {
  readMAX30102();

  float hr   = (hrValid && hrVal   > 40 && hrVal   < 200) ? (float)hrVal   : 72.0f;
  float spo2 = (spo2Valid && spo2Val > 80 && spo2Val <= 100) ? (float)spo2Val : 98.0f;
  float temp = readDS18B20();

  hrBuf[sampleIdx]   = hr;
  spo2Buf[sampleIdx] = spo2;
  tempBuf[sampleIdx] = temp;
  sampleIdx = (sampleIdx + 1) % SAMPLE_COUNT;
  if (sampleCount < SAMPLE_COUNT) sampleCount++;

  // Ortalama hesapla
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

  Serial.printf("[Ölçüm %d/%d] HR=%.0f SpO2=%.0f Temp=%.1f\n",
    sampleCount, SAMPLE_COUNT, lastHR, lastSpO2, lastTemp);
}

// ============================================================
// WEB SUNUCU — DOSYA SERVE
// ============================================================
void serveFile(const char* path, const char* mime) {
  if (!SPIFFS.exists(path)) {
    server.send(404, "text/plain", "Dosya bulunamadi");
    return;
  }
  File f = SPIFFS.open(path, "r");
  server.streamFile(f, mime);
  f.close();
}

// ============================================================
// WEB SUNUCU — /sensors ENDPOİNT
// (Kullanıcı buradan yaş/cinsiyet/boy/kilo da gönderiyor)
// ============================================================
void handleSensors() {
  server.sendHeader("Access-Control-Allow-Origin", "*");

  // Sensör hazır değilse bekliyor mesajı
  if (!sensorReady) {
    int remaining = SAMPLE_COUNT - sampleCount;
    server.send(200, "application/json",
      "{\"ready\":false,\"remaining\":" + String(remaining) + "}");
    return;
  }

  // URL parametrelerinden kullanıcı bilgisi al
  float age    = server.hasArg("age")    ? server.arg("age").toFloat()    : 0;
  float gender = server.hasArg("gender") ? server.arg("gender").toFloat() : 0;
  float weight = server.hasArg("weight") ? server.arg("weight").toFloat() : 0;
  float height = server.hasArg("height") ? server.arg("height").toFloat() : 0;

  float risk = -1;
  float confidence = 0;

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
    json += "\"risk\":"       + String(risk, 4) + ",";
    json += "\"isRisky\":"    + String(risk >= 0.5 ? "true" : "false") + ",";
    json += "\"confidence\":" + String(confidence, 1);
  } else {
    json += "\"risk\":null,\"isRisky\":null,\"confidence\":null";
  }
  json += "}";

  server.send(200, "application/json", json);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== MediTarama Başlatılıyor ===");

  // SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS başlatılamadı!");
  } else {
    Serial.println("✅ SPIFFS hazır");
  }

  // WiFi AP
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  Serial.printf("✅ WiFi AP: %s\n", WIFI_SSID);
  Serial.printf("   IP: %s\n", WiFi.softAPIP().toString().c_str());

  // DS18B20
  ds18b20.begin();
  Serial.println("✅ DS18B20 hazır");

  // MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("⚠️ MAX30102 bulunamadı — simüle değer kullanılacak");
  } else {
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    Serial.println("✅ MAX30102 hazır");
  }

  // TFLite
  initTFLite();

  // Web sunucu rotaları
  server.on("/", []() { serveFile("/index.html", "text/html"); });
  server.on("/sensors", handleSensors);
  server.onNotFound([]() {
    String path = server.uri();
    if (SPIFFS.exists(path)) {
      String mime = "text/plain";
      if (path.endsWith(".html")) mime = "text/html";
      else if (path.endsWith(".css")) mime = "text/css";
      else if (path.endsWith(".js"))  mime = "application/javascript";
      serveFile(path.c_str(), mime.c_str());
    } else {
      server.send(404, "text/plain", "Bulunamadı");
    }
  });

  server.begin();
  Serial.println("✅ Web sunucu başladı");
  Serial.println("=== Hazır! 192.168.4.1 adresine bağlan ===\n");
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
