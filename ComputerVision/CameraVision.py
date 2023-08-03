import cv2
import easyocr

def initialize_tracker(frame, bbox):
    # Initialize the KCF tracker
    tracker = cv2.TrackerKCF_create()
    # Initialize the tracker with the bounding box
    tracker.init(frame, bbox)
    return tracker

# Assuming you have a function detect_color_box(frame, color_range) that returns the bounding box (bbox) of the detected box
# ...

def detect_red_box(frame, color_range):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = color_range[0]
    upper_color = color_range[1]
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = frame[y:y+h, x:x+w]
        reader = easyocr.Reader(['en'])
        text = reader.readtext(cropped_image, detail=0)
        return [x, y, w, h], text

    return None



# Initialize your video capture (use your preferred method for reading video frames)
video_capture = cv2.VideoCapture(0)  # Change 0 to the video file path if using a file

# Read the first frame
ret, frame = video_capture.read()
if not ret:
    raise Exception("Error reading video stream")

# Detect the initial bounding box of the red box
color_range = [(0, 120, 70), (10, 255, 255)]  # Change this to the appropriate range for red color
bbox, _ = detect_red_box(frame, color_range)

# Initialize the tracker
tracker = initialize_tracker(frame, tuple(bbox))

# Start the video loop
while True:
    # Read a new frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)

    if success:
        # Draw the tracked bounding box on the frame
        x, y, w, h = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame with the tracked bounding box
    cv2.imshow("Tracking", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
