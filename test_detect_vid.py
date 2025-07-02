import cv2
import numpy as np
import time
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

class SimpleTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.last_positions = {}
        self.history = {}

    def update(self, detections):
        updated_tracks = {}

        for det in detections:
            x1, y1, x2, y2, label = det
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            min_dist = float('inf')
            matched_id = None
            for track_id, (px, py) in self.last_positions.items():
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([px, py]))
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_id = track_id

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_tracks[matched_id] = (cx, cy)
            if matched_id not in self.history:
                self.history[matched_id] = []
            self.history[matched_id].append((time.time(), (cx, cy)))

        self.last_positions = updated_tracks.copy()
        return updated_tracks

    def is_flare(self, track_id):
        history = self.history.get(track_id, [])
        if len(history) < 4:
            return False
        times, positions = zip(*history[-4:])
        speeds = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) / (times[i+1] - times[i] + 1e-6) for i in range(len(positions) - 1)]
        variance = np.var(speeds)
        avg_speed = np.mean(speeds)
        return avg_speed > 100 and variance < 5

def get_object_colors():
    """Define professional color scheme for different object types"""
    return {
        "Aircraft": (0, 255, 0),      # Green
        "Helicopter": (255, 165, 0),   # Orange
        "Missile": (0, 0, 255),        # Red
        "Drone": (255, 255, 0),        # Cyan
        "Armed": (0, 0, 255),          # Red for armed
        "Disarmed": (0, 255, 0),       # Green for disarmed
        "Default": (128, 128, 128)     # Gray for unknown
    }

def draw_professional_bbox(frame, x1, y1, x2, y2, object_type, track_id, speed, armed_status, confidence=None):
    """Draw professional-looking bounding box with enhanced styling"""
    colors = get_object_colors()
    
    if object_type in colors:
        primary_color = colors[object_type]
    else:
        primary_color = colors["Default"]
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), primary_color, 3)
    
    corner_length = min(20, box_width // 4, box_height // 4)
    corner_thickness = 4
    
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), primary_color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), primary_color, corner_thickness)
    
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), primary_color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), primary_color, corner_thickness)
    
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), primary_color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), primary_color, corner_thickness)
    
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), primary_color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), primary_color, corner_thickness)
    
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    crosshair_size = 8
    cv2.line(frame, (center_x - crosshair_size, center_y), 
             (center_x + crosshair_size, center_y), primary_color, 2)
    cv2.line(frame, (center_x, center_y - crosshair_size), 
             (center_x, center_y + crosshair_size), primary_color, 2)
    
    panel_height = 90
    panel_width = max(220, len(object_type) * 12)
    
    if y1 > panel_height + 10:
        panel_y = y1 - panel_height - 5
    else:
        panel_y = y2 + 5
    
    panel_x = max(0, min(x1, frame.shape[1] - panel_width))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  primary_color, 2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 15
    
    header_text = f"{object_type} [ID: {track_id:03d}]"
    cv2.putText(frame, header_text, (panel_x + 8, panel_y + 18),
                font, font_scale, (255, 255, 255), font_thickness)
    
    speed_text = f"Speed: {speed:.1f} km/h"
    cv2.putText(frame, speed_text, (panel_x + 8, panel_y + 18 + line_height),
                font, font_scale, (255, 255, 255), font_thickness)
    
    armed_color = colors["Armed"] if armed_status == "Armed" else colors["Disarmed"]
    status_text = f"Status: {armed_status.upper()}"
    cv2.putText(frame, status_text, (panel_x + 8, panel_y + 18 + 2*line_height),
                font, font_scale, armed_color, font_thickness)
    
    coord_text = f"Pos: ({center_x}, {center_y})"
    cv2.putText(frame, coord_text, (panel_x + 8, panel_y + 18 + 3*line_height),
                font, font_scale, (200, 200, 200), font_thickness)
    
    if confidence is not None:
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(frame, conf_text, (panel_x + 8, panel_y + 18 + 4*line_height),
                    font, font_scale, (200, 200, 200), font_thickness)

def estimate_z(x, y):
    return 1200.0

def calculate_speed(pos1, pos2, dt):
    if dt == 0:
        return 0.0
    pixel_to_meter = 0.002
    dx = (pos2[0] - pos1[0]) * pixel_to_meter
    dy = (pos2[1] - pos1[1]) * pixel_to_meter
    dz = (pos2[2] - pos1[2]) * pixel_to_meter
    dist_m = np.sqrt(dx**2 + dy**2 + dz**2)
    return (dist_m / dt) * 3.6

def is_armed(label):
    weapons = ["gun", "rifle", "pistol", "missile", "bomb", "grenade"]
    return any(weapon in label.lower() for weapon in weapons)

def classify_object_type(label):
    label_lower = label.lower()
    if label_lower in ["airplane", "aircraft", "jet", "fighter", "plane"]:
        return "Aircraft"
    elif label_lower in ["helicopter", "chopper"]:
        return "Helicopter"
    elif label_lower in ["missile", "rocket"]:
        return "Missile"
    elif label_lower in ["drone", "uav", "quadcopter"]:
        return "Drone"
    else:
        return f"{label}"

# Initialize model and video capture
model = YOLO('yolov8m.pt')
vid_path = 'plane_footage.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(vid_path)
tracker = SimpleTracker()
prev_positions = {}
prev_times = {}

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = 1280  # Match resized frame width
frame_height = 720  # Match resized frame height

# Initialize video writer
output_path = 'output_plane_footage.mp4'  # Output file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Video Info: {frame_width}x{frame_height} @ {fps}fps")
print(f"Output will be saved to: {output_path}")
print("Professional Object Detection System Started")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break
    
    # Resize frame for consistent processing
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Add timestamp overlay
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(frame, timestamp, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Run detection
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls.item())
        confidence = float(box.conf.item())
        label = model.names[cls]
        detections.append((x1, y1, x2, y2, label, confidence))

    # Update tracker
    tracked = tracker.update([(d[0], d[1], d[2], d[3], d[4]) for d in detections])

    # Draw professional bounding boxes
    for track_id, (cx, cy) in tracked.items():
        matched_det = None
        for det in detections:
            x1, y1, x2, y2, dlabel, conf = det
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matched_det = det
                break

        if matched_det is None:
            continue

        if tracker.is_flare(track_id):
            continue

        x1, y1, x2, y2, label, confidence = matched_det
        z = estimate_z(cx, cy)
        pos3d = (cx, cy, z)
        now = time.time()
        speed = 0.0

        if track_id in prev_positions:
            dt = now - prev_times[track_id]
            speed = calculate_speed(prev_positions[track_id], pos3d, dt)

        prev_positions[track_id] = pos3d
        prev_times[track_id] = now

        object_type = classify_object_type(label)
        armed_status = "Armed" if is_armed(label.lower()) else "Disarmed"

        draw_professional_bbox(frame, x1, y1, x2, y2, object_type, 
                             track_id, speed, armed_status, confidence)

        print(f"[{timestamp}] ID:{track_id:03d} | {object_type} | "
              f"Speed: {speed:.1f} km/h | {armed_status} | "
              f"Confidence: {confidence:.2f}")

    # Add system status overlay
    status_text = f"Objects Tracked: {len(tracked)} | System: ACTIVE"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("Professional Object Detection System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
print(f"Output video saved to: {output_path}")
print("System shutdown complete.")