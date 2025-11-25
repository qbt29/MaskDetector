
import cv2
import numpy as np
import joblib
import os
import sys
import warnings
from typing import List, Tuple, Optional, Dict, Any

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

sys.path.append(os.path.dirname(__file__))

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


# ======================
# CONFIG
# ======================
class Config:
    # Model
    MODEL_PATHS = [
        'mask_classifier_super_improved.pkl',
        'mask_classifier_fixed.pkl',
        'mask_classifier_augmented.pkl',
    ]
    
    # Feature extraction (должно совпадать с трейном)
    INPUT_SIZE = (100, 100)
    HIST_BINS = 12
    HSV_H_BINS = 10
    HSV_S_BINS = 10
    
    # FaceMesh
    MAX_NUM_FACES = 10
    MIN_DETECTION_CONF = 0.5
    MIN_TRACKING_CONF = 0.5
    BBOX_PADDING_RATIO = 0.1
    
    # Tracking
    TRACK_MAX_AGE = 15
    TRACK_IOU_THRESHOLD = 0.25
    
    # Drawing
    LABELS = ['Mask OK', 'No Mask', 'Wrong Mask']
    COLORS = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]
    ERROR_COLOR = (255, 255, 0)


# ======================
# FEATURE EXTRACTOR — строго как в трейне
# ======================
class FeatureExtractor:
    """Извлекает признаки точно так же, как при обучении."""
    
    def __init__(self, config: Config):
        self.config = config

    def split_into_thirds(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = img.shape[0]
        return (
            img[:h//3, :],
            img[h//3:2*h//3, :],
            img[2*h//3:, :]
        )

    def compute_hist(self, region: np.ndarray, channels: List[int], bins: List[int]) -> np.ndarray:
        hist = []
        for ch, b in zip(channels, bins):
            hch = cv2.calcHist([region], [ch], None, [b], [0, 256])
            hch = cv2.normalize(hch, hch).flatten()
            hist.append(hch)
        return np.hstack(hist)

    def extract(self, face_roi: np.ndarray) -> np.ndarray:
        """Возвращает (1, n_features) вектор признаков."""
        face_resized = cv2.resize(face_roi, self.config.INPUT_SIZE)
        top, middle, bottom = self.split_into_thirds(face_resized)
        
        # Гистограммы BGR
        hist_top = self.compute_hist(top, [0, 1, 2], [self.config.HIST_BINS] * 3)
        hist_bottom = self.compute_hist(bottom, [0, 1, 2], [self.config.HIST_BINS] * 3)
        
        # HSV для верхней трети
        hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_top], [0], None, [self.config.HSV_H_BINS], [0, 180])
        hist_s = cv2.calcHist([hsv_top], [1], None, [self.config.HSV_S_BINS], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        
        # Средние цвета
        avg_top = np.mean(top, axis=(0, 1))
        avg_middle = np.mean(middle, axis=(0, 1))
        avg_bottom = np.mean(bottom, axis=(0, 1))
        
        # Разницы
        diff_tb = np.abs(avg_top - avg_bottom)
        diff_tm = np.abs(avg_top - avg_middle)
        diff_mb = np.abs(avg_middle - avg_bottom)
        
        # Текстура и яркость
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        lap_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.var(sobelx) + np.var(sobely)
        
        brightness_top = np.mean(gray[:gray.shape[0]//3, :])
        brightness_bottom = np.mean(gray[2*gray.shape[0]//3:, :])
        brightness_contrast = abs(brightness_top - brightness_bottom)
        
        # Сборка признаков
        features = np.hstack([
            hist_top, hist_bottom,
            hist_h, hist_s,
            avg_top, avg_middle, avg_bottom,
            diff_tb, diff_tm, diff_mb,
            [lap_var, sobel_var, brightness_contrast]
        ])
        return features.reshape(1, -1)


# ======================
# CLASSIFIER WRAPPER
# ======================
class MaskClassifier:
    """Обёртка над ML-моделью."""
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        probas = self.model.predict_proba(features)[0]
        pred = int(np.argmax(probas))
        conf = float(probas[pred])
        return pred, conf


# ======================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ
# ======================
def load_mask_classifier() -> Optional[MaskClassifier]:
    """Загружает первую доступную модель классификатора масок."""
    for path in Config.MODEL_PATHS:
        if os.path.exists(path):
            try:
                clf = MaskClassifier(path)
                print(f"ML-модель загружена: '{path}'")
                return clf
            except Exception as e:
                print(f"Ошибка загрузки '{path}': {e}")
    print("ML-модель не найдена.")
    return None


# ======================
# TRACKER (TTL + IoU)
# ======================
class FaceTracker:
    """Простой трекер на основе кэша, TTL и IoU."""
    
    def __init__(self, max_age: int, iou_threshold: float):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
        self.frame_counter = 0

    def iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        x1a, y1a, w1a, h1a = boxA
        x1b, y1b, w1b, h1b = boxB
        
        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x1a + w1a, x1b + w1b)
        yi2 = min(y1a + h1a, y1b + h1b)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        areaA = w1a * h1a
        areaB = w1b * h1b
        union = areaA + areaB - inter
        
        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], int, float]]) -> Dict[int, Dict]:
        self.frame_counter += 1
        
        # Удаление устаревших треков
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if self.frame_counter - t['last_frame'] <= self.max_age
        }
        
        matched = [False] * len(detections)
        
        # Сопоставление существующих треков с новыми детекциями
        for tid, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for i, (bbox, _, _) in enumerate(detections):
                if matched[i]:
                    continue
                iou_val = self.iou(track['bbox'], bbox)
                if iou_val > best_iou and iou_val >= self.iou_threshold:
                    best_iou = iou_val
                    best_idx = i
            if best_idx != -1:
                bbox, pred, conf = detections[best_idx]
                self.tracks[tid].update({
                    'bbox': bbox,
                    'pred': pred,
                    'conf': conf,
                    'last_frame': self.frame_counter
                })
                matched[best_idx] = True
        
        # Добавление новых треков
        for i, (bbox, pred, conf) in enumerate(detections):
            if not matched[i]:
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'pred': pred,
                    'conf': conf,
                    'last_frame': self.frame_counter
                }
                self.next_id += 1
        
        return self.tracks


# ======================
# DETECTOR (MediaPipe)
# ======================
class FaceDetector:
    """Инкапсуляция MediaPipe FaceMesh."""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=self.config.MIN_DETECTION_CONF,
            min_tracking_confidence=self.config.MIN_TRACKING_CONF
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        bboxes = []
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            for flm in results.multi_face_landmarks:
                xs = [lm.x * w for lm in flm.landmark]
                ys = [lm.y * h for lm in flm.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                pad_w = int(self.config.BBOX_PADDING_RATIO * (x2 - x1))
                pad_h = int(self.config.BBOX_PADDING_RATIO * (y2 - y1))
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)
                
                bboxes.append((x1, y1, x2 - x1, y2 - y1))
        return bboxes

    def release(self):
        self.face_mesh.close()


# ======================
# DRAWING UTILS
# ======================
class Visualizer:
    """Отрисовка аннотаций и статистики."""
    
    def __init__(self, config: Config):
        self.config = config

    def draw_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  pred: int, conf: float, track_id: int, age: int):
        x, y, w, h = bbox
        
        if pred == -1:
            color = self.config.ERROR_COLOR
            label = "Error"
        else:
            color = self.config.COLORS[pred]
            label = f"{self.config.LABELS[pred]} ({conf:.2f})"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        txt_color = (255, 255, 255) if np.mean(color) < 128 else (0, 0, 0)
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)
        cv2.putText(frame, f"ID:{track_id} ({age})", (x, y + h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def draw_stats(self, frame: np.ndarray, tracked: int, detected: int):
        cv2.putText(frame, f"Tracked: {tracked}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected now: {detected}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ======================
# MAIN
# ======================
def main():
    print("=== MULTI-PERSON MASK DETECTOR (modular, DRY, SRP) ===")
    print("Модульный дизайн | Трекинг | Точно как в трейне")
    print("Press ESC to exit\n")

    classifier = load_mask_classifier()
    if classifier is None:
        return

    config = Config()
    feature_extractor = FeatureExtractor(config)
    detector = FaceDetector(config)
    tracker = FaceTracker(config.TRACK_MAX_AGE, config.TRACK_IOU_THRESHOLD)
    visualizer = Visualizer(config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Камера недоступна")
        return
    
    print("Запуск детекции...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция
            bboxes = detector.detect(frame)
            
            # Классификация
            detections = []
            for bbox in bboxes:
                x, y, w, h = bbox
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                try:
                    features = feature_extractor.extract(face_roi)
                    pred, conf = classifier.predict(features)
                    detections.append((bbox, pred, conf))
                except Exception as e:
                    print(f"ML error: {e}")
                    detections.append((bbox, -1, 0.0))
            
            # Трекинг
            tracks = tracker.update(detections)
            
            # Визуализация
            for tid, track in tracks.items():
                age = tracker.frame_counter - track['last_frame']
                visualizer.draw_face(
                    frame, track['bbox'], track['pred'], track['conf'],
                    tid, age
                )
            visualizer.draw_stats(frame, len(tracks), len(detections))
            
            cv2.imshow('Mask Detector (modular)', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("\nДетекция завершена")


if __name__ == "__main__":
    main()
