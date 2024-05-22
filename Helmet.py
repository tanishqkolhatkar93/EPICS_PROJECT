from time import sleep
import cv2 as cv
import numpy as np

frame_count = 0  # Counter for frames with detected helmets

# Initialize parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

# Load class names
classesFile = "C://Users//HP//Desktop//VITB//Winter sem 2024//Traffic-Rules-Violation-detection-system-main_rs//Traffic Rules Violation Detection System//obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfiguration = "C://Users//HP//Desktop//VITB//Winter sem 2024//Traffic-Rules-Violation-detection-system-main_rs//Traffic Rules Violation Detection System//yolov3-obj.cfg"
modelWeights = "C://Users//HP//Desktop//VITB//Winter sem 2024//Traffic-Rules-Violation-detection-system-main_rs//Traffic Rules Violation Detection System//yolov3-obj_2400.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]



# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    global frame_count
    # Draw a bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    
    if label.split(':')[0] == 'Helmet':  # Check if the detected object is a helmet
        frame_count += 1

    # Display the label and rectangle
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# Remove low-confidence bounding boxes using non-maximum suppression
def postprocess(frame, outs):
    global frame_count
    frame_count = 0  # Reset frame count for each frame
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non-maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # Draw bounding boxes and count helmets
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

# Process video frames
winName = 'Helmet Detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
vid = cv.VideoCapture('C://Users//HP//Desktop//VITB//Winter sem 2024//Traffic-Rules-Violation-detection-system-main_rs//Traffic Rules Violation Detection System//video.mp4')

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Create a blob from the frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Forward pass to get output of output layers
    outs = net.forward(getOutputsNames(net))

    # Remove low-confidence bounding boxes
    postprocess(frame, outs)

    # Display frame with bounding boxes
    cv.imshow(winName, frame)

    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
