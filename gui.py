import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

# Importing lane detection and car detection modules
# Make sure you have already defined the detect_lanes() and detect_cars() functions
# Also, ensure you have defined the suggest_lane() function for lane suggestion
from lane_detection_module import detect_lanes
from car_detection_module import detect_cars
from lane_suggestion_module import suggest_lane

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = ""

        # OpenCV video capture object
        self.vid = None

        # Create a canvas that can fit the video source
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Button to open a video file
        self.btn_open = tk.Button(window, text="Open Video", width=30, command=self.open_video)
        self.btn_open.pack(anchor=tk.CENTER, expand=True)

        # Button to start video processing
        self.btn_start = tk.Button(window, text="Start", width=30, command=self.start_video)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)

        # Button to stop video processing
        self.btn_stop = tk.Button(window, text="Stop", width=30, command=self.stop_video)
        self.btn_stop.pack(anchor=tk.CENTER, expand=True)

        # OpenCV video capture object
        self.vid = None

        # After initializing the GUI, the video is not yet playing
        self.playing = False

        # The process frame method will be called periodically
        self.process_frame()

        self.window.mainloop()

    def open_video(self):
        self.video_source = filedialog.askopenfilename()
        if self.vid is not None:
            self.vid.release()
        self.vid = cv2.VideoCapture(self.video_source)

    def start_video(self):
        self.playing = True

    def stop_video(self):
        self.playing = False

    def process_frame(self):
        if self.playing:
            ret, frame = self.vid.read()
            if ret:
                # Process the frame (lane detection, car detection, lane suggestion)
                frame_with_lanes = detect_lanes(frame)
                frame_with_cars = detect_cars(frame_with_lanes)
                # Assuming you already have the cars detected in each lane
                cars_detected = {'Lane 1': 3, 'Lane 2': 2, 'Lane 3': 4}
                suggested_lane = suggest_lane(cars_detected)
                # Display the frame with suggested lane
                cv2.putText(frame_with_cars, "Suggested Lane: " + suggested_lane, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame_with_cars, cv2.COLOR_BGR2RGB)

                # Convert the frame to ImageTk format
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

                # Update the canvas with the new frame
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Call the process_frame method after 10 ms
        self.window.after(10, self.process_frame)

# Create a window and pass it to the App class
App(tk.Tk(), "Car Detection & Lane Suggestion")
