import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# to capture feed from cam
# 0 identifies primary camera if there are multiple cameras
# second argument is optional, it is used to specify the api for video  capture
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def get_bg(image, bg):
    if bg is None:
        return image.copy().astype(float)  # initial frame
    else:
        cv2.accumulateWeighted(image, bg, .5)  # Background subtraction
        return bg


def get_gesture(image, bg):
    diff = cv2.absdiff(bg, image)  # Absolute difference b/w two images
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # Binary Thresholding of image
    return threshold, bg


def get_model():
    convnet = input_data(shape=[None, 240, 250, 1], name='input')
    # Conv Layer 1
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 2
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 3
    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 4
    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 5
    convnet = conv_2d(convnet, 256, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 6
    convnet = conv_2d(convnet, 128, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 7
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    # Conv Layer 8
    convnet = fully_connected(convnet, 1000, activation='relu')
    convnet = dropout(convnet, 0.75)
    # Fully Connected Layer with SoftMax as Activation Function
    convnet = fully_connected(convnet, 9, activation='softmax')

    # Regression for ConvNet with ADAM optimizer
    convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                         name='regression')
    model = tflearn.DNN(convnet, tensorboard_verbose=0)
    model.load('Model\\cnnModel.tfl')
    return model


def get_prediction(model):
    temp = cv2.imread('temp.png')
    gray_temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_temp.reshape(240, 250, 1)])
    confidence = np.amax(prediction) / np.sum(prediction[0])
    prediction = np.argmax(prediction)
    gesture_name = ''
    if prediction == 0:
        gesture_name = 'blank'
    elif prediction == 1:
        gesture_name = 'thumbsUp'
    elif prediction == 2:
        gesture_name = 'fist'
    elif prediction == 3:
        gesture_name = 'hello'
    elif prediction == 4:
        gesture_name = 'five'
    elif prediction == 5:
        gesture_name = 'four'
    elif prediction == 6:
        gesture_name = 'three'
    elif prediction == 7:
        gesture_name = 'two'
    elif prediction == 8:
        gesture_name = 'one'

    print(gesture_name)
    print(confidence)
    return gesture_name, confidence


if __name__ == "__main__":
    model = get_model()
    num_frames = 0  # numbers of frames
    left, top, right, bottom = 350, 30, 600, 270  # dimensions of roi
    bg = None  # initially no background
    start = False  # flag to indicate if recognition should start

    # keep taking feed until interrupted
    while True:
        # ret: flags if capture was successful {True,False}
        # frame: numpy array  which stores the feed {Each frame of video capture}
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Negate mirror view effect  of cam

        if not ret:
            print('Camera is busy')
            break
        else:
            # Draw a rectangle from top left corner to bottom right corner eg.(100, 100) to (300, 300)
            # 2 denotes the edges width
            # (0, 255, 0) is BGR color scheme
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            feed = frame.copy()
            roi = frame[top:bottom, left:right]  # region of interest i.e Hand
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Change Img to Grayscale
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Blur image to remove noise

            # Setting up background for background subtraction
            if num_frames < 25:
                bg = get_bg(blurred, bg)
                if num_frames == 1:
                    print('Setting up background')
                elif num_frames == 24:
                    print('Setup Complete')

            else:
                # foreground extraction i.e hand region  out the image
                gesture, bg = get_gesture(blurred, bg.astype('uint8'))
                if gesture is None:
                    continue
                else:
                    cv2.imshow('threshold', gesture)  # View hand gesture
                    if start:
                        cv2.imwrite('temp.png', gesture)
                        prediction, gesture_name = get_prediction(model)
            num_frames += 1

            cv2.imshow('feed', feed)  # to view an feed
            key = cv2.waitKey(1)  # waits for keyboard input for 1ms

            # breaks out of loop if 'q' is pressed
            if key == ord('q'):
                break

            # Start prediction if 's' is pressed
            if key == ord('s'):
                start = True

    cap.release()  # release camera
    cv2.destroyAllWindows()  # destroy all windows so no window maybe running in background
