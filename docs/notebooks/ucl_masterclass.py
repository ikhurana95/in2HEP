import numpy as np
import cv2
from PIL import Image
import time
from resizeimage import resizeimage
import imutils
import pickle



def save_img(img,filePath = "number.pkl"):
    """
    Writes events to pickle file. Ideally dump few objects where the objects could be any data structures
    containing other objects
    :param events:
    :param filePath:
    """
    print("Writing...")
    if filePath[-3:]!="pkl":
        filePath = filePath+".pkl"

    with open(filePath, "wb") as output:

        pickle.dump(img, output, pickle.HIGHEST_PROTOCOL)

def load(filePath,python2 = False):
    """
    Loads objects from pickle file
    :param filePath:
    :return: values in pickle file
    """
    load = []
    print("Loading...")
    with open(filePath, "rb") as file:
        hasNext = True
        if python2:

            load.append(pickle.load(file))
        else:
            load.append(pickle.load(file, encoding='latin1'))
        while hasNext:
            try:
                if python2:
                    load.append(pickle.load(file))
                else:
                    load.append(pickle.load(file, encoding='latin1'))
            except:
                hasNext = False

    if len(load) == 1:
        return load[0]
    else:
        return load


def get_webcam_img():

    """
    Creates an webcam image for the Handwritten Digit Classifier Project.

    Uses cv2 python library to apply filters on captured image so that it can
    be processed by the neural network.

    Saves a 28*28 black and white image.


    """
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Applies black and white filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Applies edge detection filter
        edged = cv2.Canny(gray,75,180)
        fgmask = fgbg.apply(edged)

        # Crops the recorded image
        w = 350
        h = 350
        x = int(gray.shape[1]/2 - w/2)
        y = int(gray.shape[0]/2 - h/2)

        crop_img = gray[y:y+h, x:x+w]
        # Image displayed without edge detection
        cv2.imshow("Press SPACE to shoot", crop_img)

        # Uncomment line below to demonstrate edge detection
        # cv2.imshow("Press SPACE to shoot", fgmask)



        # Checks if space is pressed
        if cv2.waitKey(1) & 0xFF == ord(' ') or 0xFF == ord('\n'):

            crop_img = fgmask[y:y+h, x:x+w]

            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(crop_img,kernel,iterations = 1)
            print(dilation.shape)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray  = cv2.resize(dilation,None,fx=128/350, fy=128/350, interpolation = cv2.INTER_CUBIC)
            print(gray.shape)

            gray = cv2.dilate(gray,kernel,iterations = 1)
            gray  = cv2.resize(gray,None,fx=28/128, fy=28/128, interpolation = cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            print(gray.shape)

            cv2.imwrite('digit.png',gray)

            # Saves image to a pickle file
            save_img(gray)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

    return gray
