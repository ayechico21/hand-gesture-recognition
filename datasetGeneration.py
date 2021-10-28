import cv2

# to capture feed from cam
# 0 identifies primary camera if there are multiple cameras
# second argument is optional, it is used to specify the api for video  capture
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


if __name__ == "__main__":
    num_frames = 0  # numbers of frames
    left, top, right, bottom = 350, 30, 600, 270  # dimensions of roi
    bg = None  # initially no background

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
            if num_frames < 50:
                bg = get_bg(blurred, bg)
                if num_frames == 1:
                    print('Setting up background')
                elif num_frames == 49:
                    print('Setup Complete')

            else:
                # foreground extraction i.e hand region  out the image
                gesture, bg = get_gesture(blurred, bg.astype('uint8'))
                if gesture is None:
                    continue
                else:
                    cv2.imshow('threshold', gesture)  # View hand gesture

            num_frames += 1
            cv2.imshow('feed', feed)  # to view an feed

            # waits for keyboard input for 1ms, breaks out of loop if 'q' is pressed
            if(cv2.waitKey(1) == ord('q')):
                break
    cap.release()  # release camera
    cv2.destroyAllWindows()  # destroy all windows so no window maybe running in background
