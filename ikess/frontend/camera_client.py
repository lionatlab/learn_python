import cv2
import numpy as np

from requests import get

frame_width = 28
frame_height = 28

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print('frame shpae: {}'.format(frame.shape))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    print('gray frame shpae: {}'.format(gray.shape))

    gray = cv2.resize(gray, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
    # gray = gray[np.newaxis, :, :]
    print('gray shpae: {}'.format(gray.shape))

    ret = get('http://localhost:5000/classify',
              data={'pixels': gray.tolist(),
                    'frame_size': [frame_width, frame_height]})
    print('predict: {}'.format(ret.json()))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()