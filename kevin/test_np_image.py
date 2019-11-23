import numpy as np
from PIL import Image
pre_img = Image.open('image/yoojin.jpg')
data = np.array(pre_img)
print(data)