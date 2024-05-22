import cv2

def detect_cars(frame):
    # Load the pre-trained car detection Haar Cascade classifier
    car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame

# Main function
if __name__ == "__main__":
    # Open the video file
    cap = cv2.VideoCapture("path/to/your/video/file.mp4")

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file")
    else:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            
            # Check if frame is empty
            if not ret:
                break

            # Detect cars in the frame
            frame_with_cars = detect_cars(frame)
            
            # Display the frame with cars detected
            cv2.imshow('Car Detection', frame_with_cars)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
