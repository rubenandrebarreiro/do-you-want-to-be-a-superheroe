# Do You Want To Be a Superheroe?

# Description:
# - A simple application for Faces and Eyes Detection,
#   using Masks/Filters inspired in Superheroe's concept for each detected component,
#   during a video capture;

# Authors:
# - Ruben Andre Barreiro


# The imported libraries

# DLib Library
import dlib as dl

# OpenCV Library
import cv2 as cv

# NumPy Library
import numpy as np

# NDImage from SciPy Library
from scipy import ndimage


# The Video Capture/Stream from the Device
video_capture = cv.VideoCapture(0)

# The Superheroe Mask in use by default
superheroe_mask = cv.imread("images/superheroe-mask-1.png", -1)

# The Frontal Face Detector from DLib
detector = dl.get_frontal_face_detector()

# The Facial Predictor to be used by DLib
predictor = dl.shape_predictor("predictors/shape_predictor_68_face_landmarks.dat")


# Function to resize an Image to a certain Width
def resize(image, width):
    ratio = float(width) / image.shape[1]
    dimensions = (width, int(image.shape[0] * ratio))

    print('Resizing the Image to the following dimensions:')
    print(dimensions)
    print('')

    image = cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

    return image


# Function to combine an Image that has a transparency Alpha Channel
def blend_transparent(face_image, superheroe_mask_image):
    overlay_image = superheroe_mask_image[:, :, :3]
    overlay_mask = superheroe_mask_image[:, :, 3:]

    background_mask = (255 - overlay_mask)

    overlay_mask = cv.cvtColor(overlay_mask, cv.COLOR_GRAY2BGR)
    background_mask = cv.cvtColor(background_mask, cv.COLOR_GRAY2BGR)

    face_part = (face_image * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_image * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# Function to find the Angle between two Points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])

    print('Calculating the angle between:')
    print(angle_1)
    print('and')
    print(angle_2)
    print('')

    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))


# Function to handle the change of a Superheroe Mask on the trackball
def change_superheroe_mask(position):

    print('Changing the current Superheroe Mask and loading the respectively components...')

    if position == 0:
        print('Loading the Superheroe Mask 1...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-1.png", 50, -0.3)
    elif position == 1:
        print('Loading the Superheroe Mask 2...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-2.png", 60, -1.1)
    elif position == 2:
        print('Loading the Superheroe Mask 3...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-3.png", 60, -0.65)
    elif position == 3:
        print('Loading the Superheroe Mask 4...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-4.png", 80, -0.6)
    elif position == 4:
        print('Loading the Superheroe Mask 5...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-5.png", 80, -1.2)
    elif position == 5:
        print('Loading the Superheroe Mask 6...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-6.png", 50, -0.3)
    elif position == 6:
        print('Loading the Superheroe Mask 7...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-7.png", 30, -1.0)
    elif position == 7:
        print('Loading the Superheroe Mask 8...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-8.png", 100, -0.6)
    elif position == 8:
        print('Loading the Superheroe Mask 9...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-9.png", 30, -0.95)
    elif position == 9:
        print('Loading the Superheroe Mask 10...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-10.png", 100, -0.5)
    elif position == 10:
        print('Loading the Superheroe Mask 11...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-11.png", 40, -1.0)
    elif position == 11:
        print('Loading the Superheroe Mask 12...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-12.png", 60, -0.4)
    elif position == 12:
        print('Loading the Superheroe Mask 13...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-13.png", 50, -1.0)
    elif position == 13:
        print('Loading the Superheroe Mask 14...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-14.png", 80, -0.35)
    elif position == 14:
        print('Loading the Superheroe Mask 15...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-15.png", 80, -0.4)
    elif position == 15:
        print('Loading the Superheroe Mask 16...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-16.png", 20, -1.0)
    elif position == 16:
        print('Loading the Superheroe Mask 17...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-17.png", 60, -0.4)
    elif position == 17:
        print('Loading the Superheroe Mask 18...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-18.png", 60, -0.45)
    elif position == 18:
        print('Loading the Superheroe Mask 19...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-19.png", 80, -0.4)
    elif position == 19:
        print('Loading the Superheroe Mask 20...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-20.png", 60, -0.4)
    elif position == 20:
        print('Loading the Superheroe Mask 21...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-21.png", 80, -0.7)
    elif position == 21:
        print('Loading the Superheroe Mask 22...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-22.png", 80, -0.65)
    elif position == 22:
        print('Loading the Superheroe Mask 23...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-23.png", 20, -1.0)
    elif position == 23:
        print('Loading the Superheroe Mask 24...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-24.png", 70, -0.9)
    elif position == 24:
        print('Loading the Superheroe Mask 25...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-25.png", 80, -0.7)
    elif position == 25:
        print('Loading the Superheroe Mask 26...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-26.png", 40, -0.8)
    elif position == 26:
        print('Loading the Superheroe Mask 27...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-27.png", 40, -0.8)
    elif position == 27:
        print('Loading the Superheroe Mask 28...')
        print('')
        apply_superheroe_mask("images/superheroe-mask-28.png", 80, -0.4)


# Function to apply the necessary changes, accordingly to the currently Superheroe Mask being used
def apply_superheroe_mask(path, dimensions, y_translation):
    global eye_right, eye_left, x, y, w, h, degree

    superheroe_mask_image = cv.imread(path, -1)

    # Start the Main Program
    while True:

        rectangle, image = video_capture.read()
        image = resize(image, 1200)
        image_copy = image.copy()
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # noinspection PyBroadException
        try:
            # Try to detect the Faces
            detections = detector(gray, 1)

            # Find a Face in the following Box Bounding Points
            print('Detecting a Face in a Box Bounding Points...')
            print('')

            for detected in detections:
                x = (detected.left() - dimensions)
                y = (detected.top() - dimensions)
                w = (detected.right() + dimensions)
                h = (detected.bottom() + dimensions)

            dl_rectangle = dl.rectangle(x, y, w, h)

            # Find the detected Facial Landmarks by the used Predictor
            print('Detecting Facial Landmarks by the used Predictor...')
            print('')

            detected_landmarks = predictor(gray, dl_rectangle).parts()

            landmarks = np.matrix([[point.x, point.y] for point in detected_landmarks])

            for index, point in enumerate(landmarks):
                position = (point[0, 0], point[0, 1])

                if index == 0:
                    eye_left = position
                elif index == 16:
                    eye_right = position

                # noinspection PyBroadException
                try:
                    # Just for debug
                    # Line for the angle between eyes
                    # cv.line(img_copy, eye_left, eye_right, color=(0, 255, 255))
                    degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

                except:
                    pass

            # Translate Facial Object Based on Input Object,
            # using the center point between the two detected eyes
            eye_center = (eye_left[1] + eye_right[1]) / 2

            # Translation of the Superhero Mask,
            # moving 0.3 units up
            superheroe_mask_translated = int(y_translation * (eye_center - y))

            # Resize the Superheroe Mask to width of Face and Blend image
            face_width = (w - x)

            # Resize the Superheroe Mask
            superheroe_mask_resize = resize(superheroe_mask_image, face_width)

            # Rotate Superheroe Mask based on Angle between eyes
            y_g, x_g, c_g = superheroe_mask_resize.shape
            superheroe_mask_resize_rotated = ndimage.rotate(superheroe_mask_resize, (degree + 90))
            superheroe_mask_rectangle_rotated = ndimage.rotate(image[y + superheroe_mask_translated:y + y_g + superheroe_mask_translated, x:w], (degree + 90))

            # Blending with Rotation, in Superheroe Mask
            h5, w5, s5 = superheroe_mask_rectangle_rotated.shape
            rectangle_resize = image_copy[y + superheroe_mask_translated:y + h5 + superheroe_mask_translated, x:x + w5]
            blend_mask3 = blend_transparent(rectangle_resize, superheroe_mask_resize_rotated)
            image_copy[y + superheroe_mask_translated:y + h5 + superheroe_mask_translated, x:x + w5] = blend_mask3

            cv.imshow('Do You Want To Be a Superheroe?', image_copy)

        except:
            cv.imshow('Do You Want To Be a Superheroe?', image_copy)

        if cv.waitKey(10) == 27:
            break


print('')
print('Welcome to the "Do You Want To Be a Superheroe?" application!!!')
print('')

cv.namedWindow('Do You Want To Be a Superheroe?', cv.WINDOW_GUI_EXPANDED | cv.WINDOW_AUTOSIZE)
cv.createTrackbar('Select a Superheroe Mask', 'Do You Want To Be a Superheroe?', 0, 27, change_superheroe_mask)

print('Loading the Superheroe Mask 1...')
print('')
apply_superheroe_mask("images/superheroe-mask-1.png", 50, -0.3)
