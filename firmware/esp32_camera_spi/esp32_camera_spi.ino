// ============================================================================
// esp32_camera_spi.ino - ESP32-CAM (AI-Thinker, GC2145) -> Nexys4 DDR FPGA over SPI (Pmod JA)
//
// Captures raw GRAYSCALE QVGA (320x240), averages 2x2 blocks to 160x120 with
// (a+b+c+d+2)>>2 (identical to cv2.INTER_AREA, so the FPGA sees exactly what
// seg_unet.preprocess() gives the PC), and sends one SPI burst per frame:
//
//   'C','A','M','1' | frame_no u16 | 19200 pixels (row-major) | sum16(pixels) u16   (little-endian)
//
// SPI mode 0, MSB first, SPI_HZ (default 8 MHz). The FPGA is the slave.
//
// Wiring (3.3 V logic on both sides, short wires < 20 cm):
//   ESP32-CAM GPIO15 (CS)   -> Nexys4 JA1
//   ESP32-CAM GPIO13 (MOSI) -> Nexys4 JA2
//   ESP32-CAM GPIO14 (SCLK) -> Nexys4 JA4
//   ESP32-CAM GND           -> Nexys4 JA5 (GND)
//   (JA3 / MISO not used. Do NOT connect Pmod VCC to the ESP32-CAM; keep it USB/5 V powered.)
//   GPIO13/14/15 are the SD-card pins: do not insert an SD card.
//
// Arduino IDE: Board "AI Thinker ESP32-CAM", PSRAM Enabled, Flash Mode DIO, Flash Frequency 40 MHz
//   (QIO crashes at boot on this board: "assert failed: ... init_flash").
// Tested 11 Nov 2026 on the user's board: GC2145 (PID 0x2145), GRAYSCALE, 14 fps. The first JPEG init
// failing with 0x106 and "GPIO isr service already installed" are expected messages.
// Upload: GPIO0 -> GND, reset, upload; then remove GPIO0-GND and reset to run.
// Serial (115200) prints fps once per second.
// ============================================================================
#include "esp_camera.h"
#include "esp_log.h"
#include <SPI.h>

#define SPI_HZ   8000000        // lower to 4000000 if the FPGA LED6 (camera packet error) lights
#define PIN_SCK  14
#define PIN_MOSI 13
#define PIN_CS   15

// AI-Thinker ESP32-CAM camera pins
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static const int OW = 160, OH = 120;
static uint8_t pkt[4 + 2 + OW * OH + 2];
SPIClass spi(HSPI);

static int  y_step = 1;                    // 1 = GRAYSCALE frame, 2 = YUV422 frame (Y = every 2nd byte)
static bool cam_ok = false;

static esp_err_t cam_init(pixformat_t fmt) {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM; c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = fmt;
  c.frame_size = FRAMESIZE_QVGA;            // 320x240, same as the PC pipeline
  c.jpeg_quality = 12;
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&c);
  Serial.printf("  init %-9s -> 0x%x (%s)\n",
                fmt == PIXFORMAT_GRAYSCALE ? "GRAYSCALE" : fmt == PIXFORMAT_JPEG ? "JPEG" : "YUV422",
                err, esp_err_to_name(err));
  return err;
}

// Same order as the working USB sketch (1_esp32cam/esp32_final_uart): JPEG first, then raw
// GRAYSCALE if the sensor has no hardware JPEG (GC2145). YUV422 (Y channel) is the last resort.
static bool start_camera() {
  for (int attempt = 1; attempt <= 3; attempt++) {
    Serial.printf("camera attempt %d\n", attempt);
    esp_err_t err = cam_init(PIXFORMAT_JPEG);
    esp_camera_deinit();                              // ALWAYS release the driver before the next init
    delay(100);                                       // (exactly what the working USB sketch does)
    if (err != ESP_ERR_NOT_FOUND) {
      if (cam_init(PIXFORMAT_GRAYSCALE) == ESP_OK) { y_step = 1; return true; }
      esp_camera_deinit(); delay(100);
      if (cam_init(PIXFORMAT_YUV422) == ESP_OK)    { y_step = 2; return true; }
      esp_camera_deinit(); delay(100);
    }
    delay(500);
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\nESP32-CAM -> FPGA (SPI) 160x120 grey");
  Serial.printf("chip %s, PSRAM %s (%u bytes)\n", ESP.getChipModel(), psramFound() ? "found" : "NOT FOUND",
                ESP.getPsramSize());
  cam_ok = start_camera();
  if (!cam_ok) {
    while (true) {
      Serial.println("ERR camera init failed. 0x105 = no camera answered (wrong board on this COM port, or"
                     " ribbon cable loose); 0x106 = camera not supported; 0x101 = no memory (PSRAM: Enabled?).");
      delay(2000);
    }
  }
  sensor_t *sn = esp_camera_sensor_get();
  Serial.printf("camera OK: sensor PID 0x%x, %s\n", sn->id.PID, y_step == 1 ? "GRAYSCALE" : "YUV422 (using Y)");

  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  spi.begin(PIN_SCK, -1, PIN_MOSI, -1);     // no MISO (GPIO12 is a strapping pin)
  pkt[0] = 'C'; pkt[1] = 'A'; pkt[2] = 'M'; pkt[3] = '1';
}

void loop() {
  static uint16_t frame_no = 0;
  static uint32_t n = 0, t0 = millis();
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  if (fb->width != 320 || fb->height != 240 || fb->len < (size_t)(320 * 240 * y_step)) {
    Serial.printf("unexpected frame %ux%u fmt %d\n", fb->width, fb->height, fb->format);
    esp_camera_fb_return(fb);
    delay(500);
    return;
  }
  // 2x2 average -> 160x120  (a+b+c+d+2)>>2 == cv2.INTER_AREA for exactly 2x
  uint8_t *dst = pkt + 6;
  uint16_t sum = 0;
  const uint8_t *src = fb->buf;
  for (int y = 0; y < OH; y++) {
    const uint8_t *r0 = src + (2 * y) * 320 * y_step, *r1 = r0 + 320 * y_step;
    for (int x = 0; x < OW; x++) {
      int a = 2 * x * y_step, b = (2 * x + 1) * y_step;
      uint8_t v = (uint8_t)((r0[a] + r0[b] + r1[a] + r1[b] + 2) >> 2);
      *dst++ = v;
      sum += v;
    }
  }
  esp_camera_fb_return(fb);
  pkt[4] = frame_no & 0xFF; pkt[5] = frame_no >> 8;
  pkt[6 + OW * OH] = sum & 0xFF; pkt[7 + OW * OH] = sum >> 8;
  frame_no++;

  spi.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW);
  spi.writeBytes(pkt, sizeof(pkt));
  digitalWrite(PIN_CS, HIGH);
  spi.endTransaction();

  n++;
  if (millis() - t0 >= 1000) {
    Serial.printf("%u fps (frame %u)\n", n, frame_no);
    n = 0; t0 = millis();
  }
}
