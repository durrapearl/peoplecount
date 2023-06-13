import cv2
import imutils
import numpy as np
import argparse
import RPi.GPIO as GPIO

LED_PIN = 18  # GPIO pin number for the LED
BUZZER_PIN = 23  # GPIO pin number for the buzzer

def countPeople(frame):
    # Load pre-trained HOG model for pedestrian detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect people in the frame
    bounding_box_cordinates, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    # Draw bounding boxes and count the number of people
    person_count = 0
    for (x, y, w, h) in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        person_count += 1

    # Display the number of people detected
    cv2.putText(frame, f'Total Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, person_count

def detectByCamera():
    video = cv2.VideoCapture(0)

    # Setup GPIO pins
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    while True:
        _, frame = video.read()

        frame = imutils.resize(frame, width=800)
        frame, person_count = countPeople(frame)

        cv2.imshow('Output', frame)

        # Turn on LED if there is detection
        if person_count > 0:
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)

        # Turn on buzzer if people detected > 5
        if person_count > 5:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup GPIO pins
    GPIO.cleanup()

    video.release()
    cv2.destroyAllWindows()

def detectByPathVideo(video_path):
    video = cv2.VideoCapture(video_path)

    # Setup GPIO pins
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        frame, person_count = countPeople(frame)

        cv2.imshow('Output', frame)

        # Turn on LED if there is detection
        if person_count > 0:
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)

        # Turn on buzzer if people detected > 5
        if person_count > 5:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup GPIO pins
    GPIO.cleanup()

    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print('Image Not Found. Please Enter a Valid Path (Full path of Image Should be Provided).')
        return

    image = imutils.resize(image, width=800)
    image, person_count = countPeople(image)

    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    args = argsParser()

    if args["video"] is not None:
        detectByPathVideo(args["video"])
    elif args["image"] is not None:
        detectByPathImage(args["image"])
    else:
        detectByCamera()
