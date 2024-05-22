import cv2
import numpy as np

def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]
    cv2.fillPoly(mask, [np.array(roi_vertices)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform to detect lines in the image
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    
    # Check if lines were detected
    if lines is not None:
        # Draw detected lines on the frame
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return frame

# Main function
if __name__ == "__main__":
    # Open the video file
    cap = cv2.VideoCapture("road_-_28287 (360p).mp4")

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

            # Detect lanes in the frame
            frame_with_lanes = detect_lanes(frame.copy())  # Passing a copy of frame
            
            # Display the frame with lanes detected
            cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
