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

def estimate_z(x, y):
    return 1200.0

def calculate_speed(pos1, pos2, dt):
    if dt == 0:
        return 0.0
    pixel_to_meter = 0.002  # more realistic scale: each pixel ~2mm
    dx = (pos2[0] - pos1[0]) * pixel_to_meter
    dy = (pos2[1] - pos1[1]) * pixel_to_meter
    dz = (pos2[2] - pos1[2]) * pixel_to_meter
    dist_m = np.sqrt(dx**2 + dy**2 + dz**2)
    return (dist_m / dt) * 3.6  # convert m/s to kph

def is_armed(label):
    weapons = ["gun", "rifle", "pistol", "missile", "bomb", "grenade"]
    return any(weapon in label.lower() for weapon in weapons)

model = YOLO('yolov10n.pt')
vid_path = 'plane_test.mp4'
cap = cv2.VideoCapture(vid_path)
tracker = SimpleTracker()
prev_positions = {}
prev_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls.item())
        label = model.names[cls]
        detections.append((x1, y1, x2, y2, label))

    tracked = tracker.update(detections)

    for track_id, (cx, cy) in tracked.items():
        matched_det = None
        for det in detections:
            x1, y1, x2, y2, dlabel = det
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matched_det = det
                break

        if matched_det is None:
            continue

        if tracker.is_flare(track_id):
            continue

        x1, y1, x2, y2, label = matched_det
        z = estimate_z(cx, cy)
        pos3d = (cx, cy, z)
        now = time.time()
        speed = 0.0

        if track_id in prev_positions:
            dt = now - prev_times[track_id]
            speed = calculate_speed(prev_positions[track_id], pos3d, dt)

        prev_positions[track_id] = pos3d
        prev_times[track_id] = now

        label_lower = label.lower()
        if label_lower in ["airplane", "aircraft", "jet", "fighter"]:
            object_type = "Aircraft"
        elif label_lower in ["helicopter"]:
            object_type = "Helicopter"
        elif label_lower in ["missile"]:
            object_type = "Missile"
        elif label_lower in ["drone"]:
            object_type = "Drone"
        else:
            object_type = f"Object : {label} (Not Aircraft)"

        armed_status = "Armed" if is_armed(label_lower) else "Disarmed"

        info = f"Object | ID:{track_id}  | Tracking\n  Pos: X={cx},Y={cy},Z={int(z)} | Speed: {speed:.3f} Kph\n  {object_type} | {armed_status}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for idx, line in enumerate(info.split("\n")):
            cv2.putText(frame, line, (x1, y1 - 10 + 15 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        print(info)

    cv2.imshow("Object & Aircraft Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
