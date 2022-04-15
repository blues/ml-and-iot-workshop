#include <EloquentTinyML.h>
#include <Adafruit_I2CDevice.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "model.h"

#define SCREEN_ADDRESS 0x3C
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

float randFloat(float min, float max)
{
  return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

void setup() {
    Serial.begin(115200);
    delay(5000);

    ml.begin(g_model);

    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS))
    {
      Serial.println(F("SSD1306 allocation failed"));
      for (;;)
        ; // Don't proceed, loop forever
    }
    Serial.println("Screen Connected");

    display.setRotation(2);
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
}

void loop() {
    float x = randFloat(0,1);

    float input[1] = { x };
    float predicted = ml.predict(input);

    Serial.print("X Value: ");
    Serial.println(x);
    Serial.print("Predicted Y: ");
    Serial.println(predicted);
    Serial.println();

    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("X: ");
    display.println(x);
    display.print("Predicted Y: ");
    display.println(predicted);
    display.display();
    delay(2000);
}