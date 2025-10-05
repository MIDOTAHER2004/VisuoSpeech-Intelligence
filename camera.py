import cv2
import threading
import time
from mtcnn import MTCNN

class CameraThread(threading.Thread):
    def __init__(self, src=0, detect_interval=10, min_confidence=0.95, min_face_size=50):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        self.detector = MTCNN()
        self.trackers = {}  # {id: tracker}
        self.next_id = 1
        self.faces = {}  # {id: bbox}
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size  # minimum face size

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            updated_faces = {}

            # Update trackers first
            remove_ids = []
            for fid, tracker in self.trackers.items():
                success, bbox = tracker.update(frame)
                if success:
                    updated_faces[fid] = tuple(map(int, bbox))
                else:
                    remove_ids.append(fid)

            # Remove failed trackers
            for fid in remove_ids:
                self.trackers.pop(fid)
                self.faces.pop(fid, None)

            # Run MTCNN detection every detect_interval frames
            if self.frame_count % self.detect_interval == 0:
                detections = self.detector.detect_faces(rgb)
                for det in detections:
                    confidence = det['confidence']
                    keypoints = det['keypoints']
                    x, y, w, h = det['box']

                    # Filter: low confidence + missing eyes + small faces
                    if confidence < self.min_confidence:
                        continue
                    if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
                        continue
                    if w < self.min_face_size or h < self.min_face_size:
                        continue

                    bbox = (x, y, w, h)

                    # Match with existing faces
                    matched_id = None
                    for fid, old_bbox in updated_faces.items():
                        if self._iou(bbox, old_bbox) > 0.5:
                            matched_id = fid
                            break

                    # Create new tracker if face is new
                    if matched_id is None:
                        matched_id = self.next_id
                        self.next_id += 1
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, tuple(bbox))
                        self.trackers[matched_id] = tracker

                    updated_faces[matched_id] = bbox

            self.faces = updated_faces

            # Draw bounding boxes
            for fid, (x, y, w, h) in self.faces.items():
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {fid}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with self.lock:
                self.frame = frame.copy()

            self.frame_count += 1
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return self.frame, self.faces

    def stop(self):
        self.running = False
        self.cap.release()

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0