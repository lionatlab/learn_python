import tensorflow as tf
import numpy as np
import cv2


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

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)

    return model


def main():
    mnist_net = prepare_net()

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print('frame shpae: {}'.format(frame.shape))

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        print('gray frame shpae: {}'.format(gray.shape))

        gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        gray = gray[np.newaxis, :, :]
        print('gray shpae: {}'.format(gray.shape))

        predict = mnist_net.predict_classes(gray)
        print('predict shpae: {}'.format(predict.shape))
        print('predict: {}'.format(predict))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
