### Keras Model to Tflite

Converting `.h5` and `.json` from your saved model into `.tflite format`.

Requirement :
1. Python 3.6
2. Tensorflow 1.9 <=
3. Keras 


How to use : 
1. Put your `.h5` and `.json` in `input` folder.
2. Run `python keras_to_tflite.py` in command line. Converted file will be in `output` folder.
3. add `--op t` for [optimized](https://www.tensorflow.org/lite/performance/post_training_quantization) `tflite` file. 
