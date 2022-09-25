#include <EloquentTinyML.h>
#include "model.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

float randFloat(float min, float max)
{
  return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

void setup() {
    Serial.begin(115200);
    delay(5000);

    ml.begin(g_model);
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

    delay(2000);
}