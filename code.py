import cv2
import numpy as np
import pandas as pd

# Load video
video_path = "D:\ai role assesment\AI Assignment video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define quadrants
quadrants = {
    1: ((0, 0), (width // 2, height // 2)),
    2: ((width // 2, 0), (width, height // 2)),
    3: ((0, height // 2), (width // 2, height)),
    4: ((width // 2, height // 2), (width, height))
}

# Function to detect balls
def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {
        'red': cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
        'blue': cv2.inRange(hsv, (110, 50, 50), (130, 255, 255)),
        'green': cv2.inRange(hsv, (50, 100, 100), (70, 255, 255)),
        'yellow': cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    }
    ball_positions = {}
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                ball_positions[color] = (cx, cy)
    return ball_positions

# Tracking and logging events
event_log = []
ball_positions_prev = {}
timestamp = 0

# Output video writer
out = cv2.VideoWriter('processed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ball_positions = detect_balls(frame)

    for color, position in ball_positions.items():
        for quadrant, ((x1, y1), (x2, y2)) in quadrants.items():
            if x1 <= position[0] <= x2 and y1 <= position[1] <= y2:
                if color in ball_positions_prev and ball_positions_prev[color] != quadrant:
                    event_log.append((timestamp, quadrant, color, 'Entry'))
                    cv2.putText(frame, f"Entry {color} at Q{quadrant} @ {timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ball_positions_prev[color] = quadrant

    timestamp += 1 / fps
    out.write(frame)

cap.release()
out.release()

# Save event log to CSV
df = pd.DataFrame(event_log, columns=['Time', 'Quadrant Number', 'Ball Colour', 'Type'])
df.to_csv('event_log.csv', index=False)

# Print message when done
print("Processing complete. Output video saved as 'processed_video.mp4' and event log saved as 'event_log.csv'.")
