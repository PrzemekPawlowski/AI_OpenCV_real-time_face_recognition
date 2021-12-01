import cv2

# Link to the datasets
# https://github.com/opencv/opencv/tree/master/data/haarcascades
# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam (parameter 0 is a default webcam)
# Also, instead of 0 you can enter the path to the .mp4 file
# e.g. webcam = cv2.VideoCapture('file.mp4')
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (Detects objects of different sizes in the input image (return list of coordinates of the rectangles))
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the faces
    # last parameter -> 2 this is a thickness of the rectangle
    for (x, y, width, height) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the image with the faces
    cv2.imshow('Face detector', frame)
    # Parameter 1 means that the frame will be refreshed with a frequency of 1 ms
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break
# Release the VideoCapture object
webcam.release()
