import cv2
import numpy as np

def ReadImageColor():
    imgFile = 'image/yoojin.jpg'
    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    cv2.namedWindow('test1',cv2.WINDOW_NORMAL)
    cv2.imshow('test' , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ReadImageColor()