/* Includes ---------------------------------------------------------------- */
#include "image_provider.h"

const uint32_t BAUD_RATE = 230400;

void setup() {
    Serial.begin(BAUD_RATE);
    // img = (uint8_t*)ps_calloc(IMG_SIZE, sizeof(uint8_t));

    if (!InitCamera()) return;
    char connect;
    do {
        Serial.println("d");
        delay(100);
        connect = Serial.read();
    } while(connect != 'd');
    Serial.println('m');
}

void loop() {
    delay(100);
    Serial.println("wait");
    while(!Serial.available());
    char read = Serial.read();
    if (read == 's') {
        GetImage();
    }
}