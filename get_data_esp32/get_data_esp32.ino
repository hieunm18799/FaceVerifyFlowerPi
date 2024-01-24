/* Includes ---------------------------------------------------------------- */
#include "image_provider.h"

const uint32_t BAUD_RATE = 230400;

void setup() {
    Serial.begin(BAUD_RATE);
    // img = (uint8_t*)ps_calloc(IMG_SIZE, sizeof(uint8_t));

    if (!InitCamera()) {
      ESP.restart();
      return;
    }
}

void loop() {
    delay(5000);
    char read;
    Serial.println("wait");
    while(!Serial.available());
    read = Serial.read();
    if (read == 's') {
        if (!GetImage()) ESP.restart();
    }
}