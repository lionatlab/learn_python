import tensorflow as tf
import cv2
import numpy as np
import math
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


# 10개의 이미지를 위한 배열을 생성
images = np.zeros((10, 784))

i = 0
for no in range(10):  # 10개의 이미지를 입력 받음

    gray = cv2.imread('image/' + str(no) + ".png", 0)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imwrite("0_" + str(no) + ".png", gray)

    (thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    cv2.imwrite("1_" + str(no) + ".png", gray)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    cv2.imwrite("2_" + str(no) + ".png", gray)

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    cv2.imwrite("3_" + str(no) + ".png", gray)

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    cv2.imwrite("4_" + str(no) + ".png", gray)

    flatten = gray.flatten() / 255.0
    images[i] = flatten

    i += 1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.load_weights('/tmp/mnist_checkpoint')


predictions = model.predict(images)

with tf.compat.v1.Session() as sess:
    print(tf.argmax(predictions, 1).eval())

