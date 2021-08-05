import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

img = image.load_img("Datasets/train/happy/7.jpg")
plt.imshow(img)
cv2.imread("Datasets/train/happy/7.jpg")