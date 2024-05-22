import cv2

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return

    # Read the first frame from the video
    ret, frame = cap.read()

    # Loop through each frame of the video
    while ret:
        # Display the frame
        cv2.imshow('Frame', frame)

        # Wait for the user to press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Read the next frame
        ret, frame = cap.read()

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    # Path to the input video file
    video_path = "path/to/your/video/file.mp4"

    # Process the video
    process_video(video_path)
