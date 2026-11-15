// ============================================================================
// esp32_tof_spi.ino - ESP32 + VL53L5CX (8x8, 10 Hz) -> Nexys4 DDR FPGA over SPI (Pmod JB)
// Sensor settings: 8x8 resolution, 10 Hz continuous ranging, I2C 400 kHz.
//
// One SPI burst per ranging frame:
//   'T','O','F','1' | frame_no u16 | 64 x u16 distance_mm | 64 x u8 target_status | sum16 u16
// Distances and statuses are in the ST driver's RAW index order (distance_mm[i]), NOT mirrored;
// the FPGA applies the zone geometry itself and accepts status 5/6/9/10/12 (like valid() below).
//
// SPI mode 0, MSB first, SPI_HZ (default 4 MHz). The FPGA is the slave.
//
// Wiring
//   VL53L5CX: SDA -> GPIO21, SCL -> GPIO22, VIN -> 3V3, GND -> GND (as before)
//   ESP32 D15 / GPIO15 (CS)   -> Nexys4 JB1
//   ESP32 D13 / GPIO13 (MOSI) -> Nexys4 JB2
//   ESP32 D14 / GPIO14 (SCLK) -> Nexys4 JB4
//   ESP32 GND                 -> Nexys4 JB5 (GND)
//   ESP32 D12                 -> leave UNCONNECTED (JB3/MISO is not used). GPIO12 is a boot strapping
//    pin: if anything pulls it high at reset the ESP32 selects 1.8 V flash and does not boot.
//   (D12-D15 are the ESP32's native HSPI pins: MISO 12, MOSI 13, SCLK 14, CS 15.)
//   (JB3 / MISO not used. Keep the ESP32 USB powered; do not connect Pmod VCC.)
//
// Library: SparkFun VL53L5CX.   Board: "ESP32 Dev Module".
// Serial (115200) still prints the 8x8 grid if SERIAL_GRID = 1 (useful for debugging in the Serial Monitor).
// ============================================================================
#include <Wire.h>
#include <SPI.h>
#include <SparkFun_VL53L5CX_Library.h>

#define SDA_PIN      21
#define SCL_PIN      22
#define I2C_HZ       400000
#define RANGING_HZ   10
#define SPI_HZ       4000000
#define PIN_SCK      14        // D14
#define PIN_MOSI     13        // D13
#define PIN_CS       15        // D15  (D12 = MISO, not used, leave unconnected)
#define SERIAL_GRID  1
#define MIN_VALID_MM 100       // readings nearer than this are cover-glass / wire crosstalk, not objects:
                               // sent with status 0 (invalid) so the FPGA ignores that zone

SparkFun_VL53L5CX imager;
VL53L5CX_ResultsData data;
static uint8_t pkt[4 + 2 + 128 + 64 + 2];

static bool valid(uint8_t st) {
  return st == 5 || st == 6 || st == 9 || st == 10 || st == 12;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\nVL53L5CX 8x8 ToF -> FPGA (SPI)");
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  SPI.begin(PIN_SCK, -1, PIN_MOSI, -1);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(I2C_HZ);
  Serial.println("Initialising sensor (uploads ~90 KB firmware, takes a few seconds)...");
  if (!imager.begin()) {
    Serial.println("ERR sensor not found at I2C 0x29. Check wiring, 3V3 power, LPn high, I2C_RST low.");
    while (true) delay(1000);
  }
  imager.setResolution(8 * 8);
  imager.setRangingFrequency(RANGING_HZ);
  imager.startRanging();
  pkt[0] = 'T'; pkt[1] = 'O'; pkt[2] = 'F'; pkt[3] = '1';
  Serial.println("Ranging started");
}

void loop() {
  static uint16_t frame = 0;
  if (!imager.isDataReady()) { delay(1); return; }
  if (!imager.getRangingData(&data)) return;

  uint16_t sum = 0;
  pkt[4] = frame & 0xFF; pkt[5] = frame >> 8;
  for (int i = 0; i < 64; i++) {
    uint16_t d = (uint16_t)data.distance_mm[i];
    if (data.distance_mm[i] < MIN_VALID_MM) data.target_status[i] = 0;   // too near: crosstalk
    uint8_t st = data.target_status[i];
    pkt[6 + 2 * i] = d & 0xFF;
    pkt[7 + 2 * i] = d >> 8;
    pkt[134 + i] = st;
    sum += (d & 0xFF) + (d >> 8) + st;
  }
  pkt[198] = sum & 0xFF; pkt[199] = sum >> 8;

  SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW);
  SPI.writeBytes(pkt, sizeof(pkt));
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();

#if SERIAL_GRID
  Serial.printf("Frame %u  (mm, --- = no valid target)\n", frame);
  for (int y = 0; y <= 8 * 7; y += 8) {
    for (int x = 7; x >= 0; x--) {
      int i = x + y;
      if (valid(data.target_status[i])) Serial.printf(" %4d", data.distance_mm[i]);
      else Serial.print("  ---");
    }
    Serial.println();
  }
  Serial.println();
#endif
  frame++;
}
