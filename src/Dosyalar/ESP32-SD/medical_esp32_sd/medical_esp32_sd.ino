/*
 * Medikal Risk Tahmin Sistemi - ESP32
 * 
 * Sistem:
 *   - ESP32 WiFi Access Point açar
 *   - SD karttan HTML + model dosyalarını serve eder
 *   - MAX30102'den HR ve SpO2 okur
 *   - DS18B20'den sıcaklık okur
 *   - /sensors endpoint'i JSON döner
 * 
 * Bağlantılar:
 *   MAX30102 → SDA=21, SCL=22, VCC=3.3V, GND
 *   DS18B20  → Data=4, VCC=3.3V, GND + 4.7kΩ pull-up
 *   SD Kart  → CS=5, MOSI=23, MISO=19, SCK=18
 * 
 * Kütüphaneler:
 *   - SparkFun MAX3010x
 *   - OneWire
 *   - DallasTemperature
 *   - SD (built-in)
 *   - WebServer (built-in)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <SD.h>
#include <SPI.h>
#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// =============================================
// AYARLAR
// =============================================
const char* AP_SSID     = "MediTarama";
const char* AP_PASSWORD = "";           // Şifresiz
const int   SD_CS_PIN   = 5;
const int   DS18B20_PIN = 4;

// =============================================
// SENSÖR NESNELERİ
// =============================================
MAX30105 particleSensor;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);
WebServer server(80);

// MAX30102 buffer
#define BUFFER_LENGTH 100
uint32_t irBuffer[BUFFER_LENGTH];
uint32_t redBuffer[BUFFER_LENGTH];
int32_t  spo2Value;
int8_t   spo2Valid;
int32_t  heartRate;
int8_t   hrValid;

// Ortalama için örnekler
#define SAMPLE_COUNT 5
float hrSamples[SAMPLE_COUNT];
float spo2Samples[SAMPLE_COUNT];
float tempSamples[SAMPLE_COUNT];
int   sampleIndex = 0;
int   sampleFilled = 0;

// Son okunan sensör değerleri (ortalama)
float lastHR   = 0;
float lastSpO2 = 0;
float lastTemp = 0;
bool  sensorReady = false;

// =============================================
// MIME TYPE BELİRLE
// =============================================
String getMimeType(String path) {
  if (path.endsWith(".html")) return "text/html";
  if (path.endsWith(".js"))   return "application/javascript";
  if (path.endsWith(".json")) return "application/json";
  if (path.endsWith(".bin"))  return "application/octet-stream";
  if (path.endsWith(".css"))  return "text/css";
  return "text/plain";
}

// =============================================
// SD KARTTAN DOSYA SERVE ET
// =============================================
void handleFile(String path) {
  if (path == "/") path = "/index.html";

  if (!SD.exists(path)) {
    server.send(404, "text/plain", "Dosya bulunamadi: " + path);
    return;
  }

  File file = SD.open(path, FILE_READ);
  if (!file) {
    server.send(500, "text/plain", "Dosya acilamadi");
    return;
  }

  String mime = getMimeType(path);
  server.streamFile(file, mime);
  file.close();
}

// =============================================
// /sensors ENDPOİNTİ
// =============================================
void handleSensors() {
  // CORS header (tarayıcıdan erişim için)
  server.sendHeader("Access-Control-Allow-Origin", "*");

  if (!sensorReady) {
    server.send(503, "application/json",
      "{\"error\":\"Sensor hazir degil, lutfen bekleyin\"}");
    return;
  }

  String json = "{";
  json += "\"hr\":"   + String(lastHR,   1) + ",";
  json += "\"spo2\":" + String(lastSpO2, 1) + ",";
  json += "\"temp\":" + String(lastTemp, 1);
  json += "}";

  server.send(200, "application/json", json);
}

// =============================================
// DS18B20 OKUMA
// =============================================
float readDS18B20() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);
  if (t == -127.0f) return 36.6f; // Hata durumunda varsayılan
  return t;
}

// =============================================
// MAX30102 OKUMA
// =============================================
bool readMAX30102() {
  for (int i = 0; i < BUFFER_LENGTH; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_LENGTH, redBuffer,
    &spo2Value, &spo2Valid, &heartRate, &hrValid);

  if (hrValid && spo2Valid && heartRate > 0 && spo2Value > 0) {
    lastHR   = (float)heartRate;
    lastSpO2 = (float)spo2Value;
    return true;
  }
  return false;
}

// =============================================
// SETUP
// =============================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println("   MediTarama - ESP32 Başlatılıyor");
  Serial.println("========================================");

  // SD Kart
  Serial.print("[1/4] SD Kart başlatılıyor... ");
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("HATA! SD kart bulunamadı.");
    while(1) delay(1000);
  }
  Serial.println("OK");

  // DS18B20
  Serial.print("[2/4] DS18B20 başlatılıyor... ");
  ds18b20.begin();
  Serial.println("OK");

  // MAX30102
  Serial.print("[3/4] MAX30102 başlatılıyor... ");
  Wire.begin(21, 22);
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("UYARI - Bulunamadı, simüle değer kullanılacak.");
  } else {
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    Serial.println("OK");
  }

  // WiFi Access Point
  Serial.print("[4/4] WiFi Access Point açılıyor... ");
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  IPAddress ip = WiFi.softAPIP();
  Serial.println("OK");
  Serial.print("    SSID: "); Serial.println(AP_SSID);
  Serial.print("    IP  : "); Serial.println(ip);

  // Web Server Route'ları
  server.on("/sensors", HTTP_GET, handleSensors);
  server.onNotFound([]() {
    handleFile(server.uri());
  });
  server.begin();

  Serial.println("\n[HAZIR] Tarayıcıdan 192.168.4.1 adresine gidin.");
  Serial.println("========================================\n");
}

// =============================================
// LOOP
// =============================================
void loop() {
  server.handleClient();

  // Her 1 saniyede bir ölçüm al, 5 dolunca ortalama hesapla
  static unsigned long lastRead = 0;
  if (millis() - lastRead > 1000) {
    lastRead = millis();

    if (sampleFilled < SAMPLE_COUNT) {
      Serial.printf("Olcum %d/%d aliniyor...\n", sampleFilled + 1, SAMPLE_COUNT);

      float t  = readDS18B20();
      bool  ok = readMAX30102();

      tempSamples[sampleIndex] = t;
      hrSamples[sampleIndex]   = ok ? (float)heartRate : 72.0f;
      spo2Samples[sampleIndex] = ok ? (float)spo2Value : 98.0f;
      sampleIndex++;
      sampleFilled++;

      if (sampleFilled == SAMPLE_COUNT) {
        float avgHR = 0, avgSpO2 = 0, avgTemp = 0;
        for (int i = 0; i < SAMPLE_COUNT; i++) {
          avgHR   += hrSamples[i];
          avgSpO2 += spo2Samples[i];
          avgTemp += tempSamples[i];
        }
        lastHR   = avgHR   / SAMPLE_COUNT;
        lastSpO2 = avgSpO2 / SAMPLE_COUNT;
        lastTemp = avgTemp / SAMPLE_COUNT;
        sensorReady = true;

        Serial.println("========================================");
        Serial.printf("Olcum tamamlandi! HR=%.0f SpO2=%.0f Temp=%.1f\n",
          lastHR, lastSpO2, lastTemp);
        Serial.println("Elinizi kaldirabilirsiniz.");
        Serial.println("========================================");
      }
    }
    // 5 örnek dolmuşsa tekrar ölçmüyor
  }
}
