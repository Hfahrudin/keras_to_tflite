import cv2
import sys
from PIL import Image
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--op", help="optimize the weight")
args = parser.parse_args()


from keras.models import model_from_json
import tensorflow as tf
from keras import optimizers



json_file = open('input/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('input/model.h5')


loaded_model.save(f'output/saved_model.h5')


converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("output/saved_model.h5")
if args.op:
    converter.post_training_quantize = True
tflite_model = converter.convert()
with open(f'output/model.tflite', 'wb') as f:
    f.write(tflite_model)

