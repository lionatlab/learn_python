import cv2 as cv
import numpy as np
from requests import put, get

img = cv.imread('../imgs/mnist_7.png')
# print('img: {}'.format(img))
assert img is not None

frame_width = 28
frame_height = 28

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
img = cv.resize(img, (frame_width, frame_height))

# cv.imshow('test', img)
# cv.waitKey(0)

frame_bytes = img.tobytes()
print('frame_bytes : {}'.format(frame_bytes))

img_list = img.tolist()
print('img_list : {}'.format(img_list))

ret = get('http://localhost:5000/classify',
          data={'pixels': img_list,
                'frame_size': [frame_width, frame_height]})
print('ret: {}'.format(ret.json()))