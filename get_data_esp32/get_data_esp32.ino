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
    Serial.println('l');
}

void loop() {
    // delay(100);
    char read;
    do {
        Serial.println("wait");
        delay(100);
        read = Serial.read();
    } while(read != 's');
    // Serial.println("wait");
    // while(!Serial.available());
    // read = Serial.read();
    // if (read == 's') {
    GetImage();
    // }
}