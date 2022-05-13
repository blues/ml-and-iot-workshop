# Lab 1 - Linear Regression with TensorFlow Lite and the Blues Wireless Swan

An end-to-end example of the process of creating MCU-friendly models. This example illustrates the complete process including:

1. [Training a simple linear regression model with TensorFlow and Keras](#training-the-model).
2. [Converting that model to the TensorFlow Lite FlatBuffer format](#converting-the-tensorflow-model-to-tflite).
3. [Converting the TFLite FlatBuffer model to a C byte array](#creating-an-mcu-friendly-representation-of-the-model).
4. [Performing inference with the model on a Blues Wireless Swan using TensorFlow Lite for Microcontrollers](#performing-inference-on-the-swan).

The end result is Machine Learning on an embedded device!

## Prerequisites

If you're planning to run the model training process locally, you'll need to install the following:

- [TensorFlow](https://www.tensorflow.org/install)
- [Numpy](https://numpy.org/install/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)

For building the project for Swan, you'll need to install the ZephyrRTOS project using the [instructions here](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)

## Training the model

For this example, we'll create a model that helps our machine learn the equation `y = mx + c`. This is, of course, a contrived example, but we've chosen something simple to keep the focus on the process of going from model to MCU, not on the model training process itself.

The steps below walk through the process of creating the model at a high-level. For detailed instructions, view the [Jupyter notebook source](/linear_regression.ipynb) or this [online Colab notebook from Google](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_lite/tflite_c01_linear_regression.ipynb).

In Keras, we can use the following few lines of Python to create our model.

```python
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=200, verbose=1)
```

Once we've trained the model (via the `fit` command), we'll want to generate a saved model file

```python
export_dir = 'saved_model/'
tf.saved_model.save(model, export_dir)
```

## Converting the TensorFlow Model to TFLite

TensorFlow provides a built-in converter for taking full TF models and outputting TFLite models that can be used in mobile and embedded devices. Assuming you have a saved model file on hand, this process is pretty simple.

```python
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)
```

Once you've created a TFLite model, you're ready to move to your Blues device!

**Note**: The [Jupyter notebook source](linear_regression.ipynb) for this example also contains instructions for testing out your TFLite model and running inference using the Python TFLite interpreter. If you're creating your own models, we recommend doing this before moving onto an MCU.

## Creating an MCU-friendly representation of the model

Many MCU's (including the Blues Swan) do not have native filesystem support, which means you'll need to load your TFLite model into memory in order to use it. The recommended way to do this is to convert your model into a C array and compile it into your project.

On Linux or macOS, you can do this with the `xxd` command. On Windows, this command is supported in bash for WSL.

```bash
xxd -i model.tflite > model.h
```

## Performing inference on The Swan

Once you have a MCU-friendly model in hand, you're ready to use it. You'll create a new PlatformIO project for this section, so make sure you've installed the PlatformIO VSCode Extension.

### Configuring TFLite for inference

1. Create a new PlatformIO Project targeting the Swan.

1. Once PlatformIO creates your project, move the `model.h` file into the `src` directory.

1. To use TFLite for Swan, you'll need the `EloquentTinML` library, which you can search for and install under the Platform IO Libraries UI.

1. Once you have installed the library, you'll want to include the following header file at the top of your project source.

```cpp
#include <EloquentTinyML.h>
```

3. Next, include the C array version of your model you created previously

```cpp
#include "model.h"
```

1. Then, specify a few defines and initialize the library.

```cpp
#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 2*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;
```

5. In the `setup` function, initialize the serial monitor and give yourself a delay to open the monitor. Then call the library `begin` method with the name of your model array variable .

```cpp
void setup() {
    Serial.begin(115200);
    delay(5000);

    ml.begin(g_model);
}
```

### Performing inference

Now for the fun-part, inference! In this example project, inference is triggered when a user button is pressed. Once triggered, the application performs inference every two seconds, passing a random `x` value between 0 and 1 into the model each time, invoking the model and obtaining the `y` result. After each run, the input (`x`) and output (`y`) values are output to the serial console.

1. The first step is to provide TFLite with an input value. I created a `randFloat` function to give me a float value between 0 and 1, and then set the result on the input tensor of the model.

```cpp
float randFloat(float min, float max)
{
  return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}
```

1. Next, we'll run our input through the model by assigning the `x` value to the input, and calling the interpreter's `predict` command.

```cpp
float x = randFloat(0,1);

float input[1] = { x };
float predicted = ml.predict(input);
```
3. Finally, we'll output the X and Predicted Y to the serial console.

```cpp
Serial.print("X Value: ");
Serial.println(x);
Serial.print("Predicted Y: ");
Serial.println(predicted);
Serial.println();
delay(2000);
```

1. Upload the program and open the monitor to see the values on the screen.

```bash
X Value: 0.51
Predicted Y: 0.12

X Value: 0.59
Predicted Y: 0.29

X Value: 0.55
Predicted Y: 0.22
```

### Bonus: Displaying X and Y values on the connected display

If you've reached this point and you have time, why not output the X and Y values to the Hammer screen? Try on your own, or use the compled project in the `linear_regression` folder for inspiration.