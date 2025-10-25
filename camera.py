import cv2
import threading
import time
from mtcnn import MTCNN

class CameraThread(threading.Thread):
    def __init__(self, src=0, detect_interval=5, min_confidence=0.95, min_face_size=50):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        self.detector = MTCNN()
        self.trackers = {}      
        self.next_id = 1
        self.faces = {}         
        self.last_seen = {}     
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            updated_faces = {}

            remove_ids = []
            for fid, tracker in list(self.trackers.items()):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    x, y = max(0, x), max(0, y)
                    updated_faces[fid] = (x, y, w, h)
                    self.last_seen[fid] = time.time()
                else:
                    
                    if fid in self.last_seen and time.time() - self.last_seen[fid] > 0.2:
                        remove_ids.append(fid)

            for fid in remove_ids:
                self.trackers.pop(fid, None)
                self.faces.pop(fid, None)
                self.last_seen.pop(fid, None)

            if self.frame_count % self.detect_interval == 0:
                small_rgb = cv2.resize(rgb, (0,0), fx=0.5, fy=0.5)
                detections = self.detector.detect_faces(small_rgb)
                for det in detections:
                    confidence = det['confidence']
                    x, y, w, h = det['box']
                    x, y, w, h = int(x*2), int(y*2), int(h*2), int(w*2)

                    if confidence < self.min_confidence or w < self.min_face_size or h < self.min_face_size:
                        continue

                    bbox = (x, y, w, h)
                    matched_id = None
                    
                    for fid, old_bbox in updated_faces.items():
                        if self._iou(bbox, old_bbox) > 0.5:
                            matched_id = fid
                            break

                    if matched_id is None:
                        matched_id = self.next_id
                        self.next_id += 1
                        try:
                            tracker = cv2.TrackerCSRT_create()
                        except AttributeError:
                            tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, tuple(bbox))
                        self.trackers[matched_id] = tracker
                        self.last_seen[matched_id] = time.time()

                    updated_faces[matched_id] = bbox

            self.faces = updated_faces

            for fid, (x, y, w, h) in self.faces.items():
                x, y = max(0, x), max(0, y)
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
        time.sleep(0.1)
        if self.cap.isOpened():
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
