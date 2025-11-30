import numpy as np
import cv2 as cv
import tensorflow as tf
import keras
IMG_SIZE = 224
model= keras.models.load_model("trained.keras")
img = cv.imread("mask.jpeg")
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img=tf.image.resize(img,(IMG_SIZE,IMG_SIZE))
img = tf.expand_dims(img, axis=0)  # Shape becomes (1, 224, 224, 3)
img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
print(model.predict(img))