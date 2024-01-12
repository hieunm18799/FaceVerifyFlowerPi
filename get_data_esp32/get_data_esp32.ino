/* Includes ---------------------------------------------------------------- */
#include "image_provider.h"

// #define ESP_NN true
const uint32_t BAUD_RATE = 460800;
const uint32_t IMG_SIZE = 1440000;
// Variables
uint8_t* img;

/**
 * @brief      Arduino setup function
 */
void setup() {
    Serial.begin(BAUD_RATE);
    // img = (uint8_t*)ps_calloc(IMG_SIZE, sizeof(uint8_t));

    if (!InitCamera()) return;
    char connect;
    do {
        Serial.println("d");
        delay(1000);
        connect = Serial.read();
    } while(connect != 'd');
    Serial.println('m');
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
    delay(1000);
    while(!Serial.available());
    uint16_t read = Serial.read();
    // Serial.println(read);
    if (read == 's') { // Train with a sample
        GetImage(1600,1200,3,img);
        // Serial.println(IMG_SIZE);
        // Serial.write((uint8_t*) &img, IMG_SIZE * sizeof(uint8_t));
    }
}