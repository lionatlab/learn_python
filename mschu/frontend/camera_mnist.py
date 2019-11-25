import tensorflow as tf
import numpy as np
import cv2
import time


def prepare_net():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5) # model training
    model.evaluate(x_test,  y_test, verbose=2) # model evaluation

    return model


def main():
    mnist_net = prepare_net()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        # shape : width, height, channel
        ret, frame = cap.read()
        print('frame shpae: {}'.format(frame.shape))

        ### Our operations on the frame come here

        # cvtColor : convert color
        # astype : convert data type from int to float32
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        print('gray frame shape: {}'.format(gray.shape))

        # resize : image shape = (28, 28)
        # 차원을 추가하는 이유는 RGB는 3차원인데, GRAY는 2차원이어서, 1차원 추가해야한다. 그래야 tensorflow에서 사용한다.
        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        gray_test = gray[np.newaxis, :, :]
        print('gray shape: {}'.format(gray.shape))

        # predict class for gray image
        predict = mnist_net.predict_classes(gray_test)
        print('predict shape: {}'.format(predict.shape))
        print('predict: {}'.format(predict))

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()