#include "image_provider.h"
#include <base64.h>

// Get the camera module ready
bool InitCamera() {
  Serial.println("Attempting to start Camera");
  
  // initialize the camera OV2640
  static camera_config_t config = {
    .pin_pwdn = 32,
    .pin_reset = -1,
    .pin_xclk = 0,
    .pin_sscb_sda = 26,
    .pin_sscb_scl = 27,

    .pin_d7 = 35,
    .pin_d6 = 34,
    .pin_d5 = 39,
    .pin_d4 = 36,
    .pin_d3 = 21,
    .pin_d2 = 19,
    .pin_d1 = 18,
    .pin_d0 = 5,
    .pin_vsync = 25,
    .pin_href = 23,
    .pin_pclk = 22,

    // XCLK 20MHz or 10MHz for OV2640 float_t FPS (Experimental)
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG, // YUV422,GRAYSCALE,RGB565,JPEG
    .frame_size = FRAMESIZE_UXGA,   // QQVGA-UXGA Do not use sizes above QVGA when not JPEG

    .jpeg_quality = 10, // 0-63 lower number means higher quality
    .fb_count = 2,      // if more than one, i2s runs in continuous mode. Use only with JPEG
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_LATEST,
  };
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK)
  {
      Serial.println("Camera init failed with error");
      return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  // s->set_hmirror(s, 1);        // 0 = disable , 1 = enable
  // s->set_vflip(s, 1);
  // s->set_gain_ctrl(s, 1);     // auto gain on
  // s->set_exposure_ctrl(s, 1); // auto exposure on
  // s->set_awb_gain(s, 1);      // Auto White Balance enable (0 or 1)
  // s->set_brightness(s, 1);    // up the brightness just a bit
  // s->set_contrast(s, 1);      // -2 to 2
  // s->set_saturation(s, -1);   // lower the saturation
  /////////////////////////////////////////////////
  s->set_brightness(s, 1);     // -2 to 2
  s->set_contrast(s, 1);       // -2 to 2
  s->set_saturation(s, -1);     // -2 to 2
  s->set_special_effect(s, 0); // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
  s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
  s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
  s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
  s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
  s->set_aec2(s, 0);           // 0 = disable , 1 = enable
  s->set_ae_level(s, 0);       // -2 to 2
  s->set_aec_value(s, 300);    // 0 to 1200
  s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
  s->set_agc_gain(s, 0);       // 0 to 30
  s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
  s->set_bpc(s, 0);            // 0 = disable , 1 = enable
  s->set_wpc(s, 1);            // 0 = disable , 1 = enable
  s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
  s->set_lenc(s, 1);           // 0 = disable , 1 = enable
  s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
  s->set_vflip(s, 0);          // 0 = disable , 1 = enable
  // s->set_hmirror(s, 1);        // 0 = disable , 1 = enable
  // s->set_vflip(s, 1);          // 0 = disable , 1 = enable
  s->set_dcw(s, 1);            // 0 = disable , 1 = enable
  s->set_colorbar(s, 0);       // 0 = disable , 1 = enable

  delay(15000);
  // for (uint8_t i = 0; i < 7; i++) {
  //   delay(100);
  //   camera_fb_t *fb = esp_camera_fb_get();

  //   if (!fb) {
  //       Serial.println("ERR: Camera capture failed during warm-up");
  //       return false;
  //   }

  //   esp_camera_fb_return(fb);
  //   }

  return true;
}

// Decode the JPEG image, crop it, and convert it to greyscale
bool DecodeAndProcessImage() {
  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) {
      Serial.println("Camera capture failed");
      return false;
  }

  // Serial.println(fb->len);
  // Serial.write(fb->buf, fb->len * sizeof(uint8_t));
  String encoded = base64::encode(fb->buf, fb->len);
  Serial.write(encoded.c_str(), encoded.length());    
  Serial.println();
  esp_camera_fb_return(fb);

  return true;
}

// Get an image from the camera module
bool GetImage() {
  bool decode_status = DecodeAndProcessImage();
  
  if (decode_status != true) {
    Serial.println("DecodeAndProcessImage failed");
    return decode_status;
  }

  return true;
}