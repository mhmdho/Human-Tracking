from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import cv2


# Load the model
model = YOLO('models\modelmp.pt')

# Open the video file
video_path = 'input\/battlefield.mp4'
# video_path = 'input\shopping.mp4'
cap = cv2.VideoCapture(video_path)

out = cv2.VideoWriter("output\/battlefield_output_posm.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (852, 480))
# out = cv2.VideoWriter("output\shoping_output_segm.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))

# Store the track history
track_history = defaultdict(lambda: [])
frame_rate = 1
count = 1


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    count += 1
    if count % frame_rate != 0:
        continue

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
        

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id != None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2 -15)), (int(x + w/2), int(y + h/2)), color=(0, 0, 230), thickness=2)
            cv2.rectangle(frame, (int(x - w/2 +5), int(y - h/2 -3)), (int(x + w/2 -5), int(y - h/2 -12)), color=(0, 0, 230), thickness=7)
            cv2.putText(frame, f'ID: {track_id}', (int(x - w/2 +6), int(y - h/2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 2)

        out.write(frame)
        # Display the annotated frame
        cv2.imshow("Human Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
