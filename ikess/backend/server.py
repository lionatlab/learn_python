import cv2 as cv
import numpy as np

from flask import Flask, request
from flask_restful import Resource, Api

from ikess.backend.mnist_net import pretrained_net

app = Flask(__name__)
api = Api(app)


def create_img(pixels, _w, _h):
    w = int(_w)
    h = int(_h)
    print('pixels length: {}'.format(len(pixels)))
    assert len(pixels) == w * h

    image = np.zeros((h * w), np.float)
    image[:] = pixels

    image = image.reshape((h, w))

    return image


class DigitClassifier(Resource):
    def get(self):
        pixels = request.form.getlist("pixels", type=float)
        frame_size = request.form.getlist('frame_size')
        frame_width = frame_size[0]
        frame_height = frame_size[1]
        print('pixels: {}'.format(pixels))
        print('frame_width: {}'.format(frame_width))
        print('frame_height: {}'.format(frame_height))

        img = create_img(pixels, frame_width, frame_height)
        cv.imwrite('img_from_client.png', img*255)

        mnist_net = pretrained_net()
        print('mnist_net: {}'.format(mnist_net))

        img = img[np.newaxis, :, :]
        predict = mnist_net.predict_classes(img)
        print('predict: {}'.format(predict))

        return {'predict': str(predict)}


api.add_resource(DigitClassifier, '/classify')


if __name__ == '__main__':
    app.run(debug=True)
