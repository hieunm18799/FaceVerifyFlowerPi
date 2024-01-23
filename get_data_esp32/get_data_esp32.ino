/* Includes ---------------------------------------------------------------- */
#include "image_provider.h"

const uint32_t BAUD_RATE = 460800;

void setup() {
    Serial.begin(BAUD_RATE);
    // img = (uint8_t*)ps_calloc(IMG_SIZE, sizeof(uint8_t));

    if (!InitCamera()) {
      ESP.restart();
      return;
    }
}

void loop() {
    // delay(100);
    char read;
    Serial.println("wait");
    while(!Serial.available());
    read = Serial.read();
    if (read == 's') {
        if (!GetImage()) ESP.restart();
    }
}