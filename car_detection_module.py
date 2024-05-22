import cv2
import numpy as np
import pandas

# Load the pre-trained car detection model
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')



# Open a video capture object (you can replace 'video.mp4' with the path to your video file or use 0 for webcam)
cap = cv2.VideoCapture('car_-_2165 (360p).mp4')

# Initialize variables for motion detection
prev_gray = None
moving_threshold = 500  # Adjust this threshold based on your environment

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('Vid.mp4', fourcc, 24, (w, h))

# Loop to process each frame
success, frame = cap.read()
while success:
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Process each detected car
    for (x, y, w, h) in cars:
        # Draw a rectangle around the detected car
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Region of Interest (ROI) for motion detection
        roi_gray = gray[y:y + h, x:x + w]

        # Motion detection
        if prev_gray is not None and prev_gray.shape == roi_gray.shape:
            # Ensure that both arrays have the same shape before performing absdiff
            delta = cv2.absdiff(prev_gray, roi_gray)
            delta_threshold = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]

            # Count the number of non-zero pixels (motion)
            motion_count = np.count_nonzero(delta_threshold)

            # Determine if the car is moving based on the motion count
            if motion_count > moving_threshold:
                color = (0, 0, 255)  # Red color for moving
                state = "Moving"
            else:
                color = (0, 255, 0)  # Green color for idle
                state = "Idle"

            # Draw a colored rectangle around the car
            cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1)
            cv2.putText(frame, state, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    # cv2_imshow(frame)
    video_out.write(frame)
    success, frame=cap.read()

    # Update the previous frame for the next iteration
    prev_gray = roi_gray

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_out.release()

cap.release()
cv2.destroyAllWindows()