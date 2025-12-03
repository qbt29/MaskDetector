import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time  
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


class MaskClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc2(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc3(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc4(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.fc5(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn5(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        x = self.output(x)
        return x


class FeatureExtractor:
    def __init__(self):
        self.input_size = (100, 100)

    def extract(self, face_roi):
        if face_roi.size == 0:
            return np.zeros((1, 119))

        try:
            face = cv2.resize(face_roi, self.input_size)
            h, w = face.shape[:2]

            top = face[:h // 3, :]
            middle = face[h // 3:2 * h // 3, :]
            bottom = face[2 * h // 3:, :]

            hist_top = self._calc_hist(top, 12)
            hist_bottom = self._calc_hist(bottom, 12)

            hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv_top], [0], None, [10], [0, 180])
            hist_s = cv2.calcHist([hsv_top], [1], None, [10], [0, 256])
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()

            avg_top = np.mean(top, axis=(0, 1))
            avg_middle = np.mean(middle, axis=(0, 1))
            avg_bottom = np.mean(bottom, axis=(0, 1))

            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            avg_lab = np.mean(lab, axis=(0, 1))

            diff_tb = np.abs(avg_top - avg_bottom)
            diff_tm = np.abs(avg_top - avg_middle)
            diff_mb = np.abs(avg_middle - avg_bottom)

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            lap_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_var = np.var(sobelx) + np.var(sobely)

            brightness_top = np.mean(gray[:h // 3, :])
            brightness_bottom = np.mean(gray[2 * h // 3:, :])
            brightness_diff = abs(brightness_top - brightness_bottom)

            gray_std = np.std(gray)
            gray_mean = np.mean(gray)
            gray_skew = np.mean((gray - gray_mean) ** 3) / (gray_std ** 3) if gray_std > 0 else 0

            aspect_ratio = h / w if w > 0 else 1.0

            features = np.hstack([
                hist_top, hist_bottom,
                hist_h, hist_s,
                avg_top, avg_middle, avg_bottom,
                avg_lab,
                diff_tb, diff_tm, diff_mb,
                [lap_var, sobel_var, brightness_diff, aspect_ratio,
                 gray_std, gray_skew]  
            ])

            if len(features) != 119:
                print(f" Исправлена размерность: {len(features)} → 119")
                if len(features) < 119:
                    features = np.pad(features, (0, 119 - len(features)), 'constant')
                else:
                    features = features[:119]

            return features.reshape(1, -1)

        except Exception as e:
            return np.zeros((1, 119))

    def _calc_hist(self, img, bins):
        hist = []
        for ch in range(3):
            h = cv2.calcHist([img], [ch], None, [bins], [0, 256])
            h = cv2.normalize(h, h).flatten()
            hist.append(h)
        return np.hstack(hist)


class FaceDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        """Возвращает список (x, y, w, h) — ТОЧНО КАК В CLI"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                padding = int((x2 - x1) * 0.15)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                faces.append((x1, y1, x2 - x1, y2 - y1)) 
        return faces

    def __del__(self):
        self.face_mesh.close()


class BBoxSmoother:
    """Полностью сохранён из CLI — сглаживание идентично"""
    def __init__(self, alpha=0.8, max_age=10, min_iou=0.3):
        self.alpha = alpha
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks = {}
        self.next_id = 0
        self.colors = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]
        self.labels = ['Mask OK', 'No Mask', 'Wrong Mask']

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections, predictions, confidences):
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]

        matched_tracks = set()
        matched_detections = set()

        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue

            best_iou = 0
            best_det_idx = -1

            for det_idx, det_box in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                iou = self._calculate_iou(track['bbox'], det_box)
                if iou > best_iou and iou > self.min_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx != -1:
                det_box = detections[best_det_idx]
                smoothed_box = (
                    int(self.alpha * track['bbox'][0] + (1 - self.alpha) * det_box[0]),
                    int(self.alpha * track['bbox'][1] + (1 - self.alpha) * det_box[1]),
                    int(self.alpha * track['bbox'][2] + (1 - self.alpha) * det_box[2]),
                    int(self.alpha * track['bbox'][3] + (1 - self.alpha) * det_box[3])
                )
                self.tracks[track_id]['bbox'] = smoothed_box
                self.tracks[track_id]['pred'] = predictions[best_det_idx]
                self.tracks[track_id]['conf'] = confidences[best_det_idx]
                self.tracks[track_id]['age'] = 0
                color_idx = predictions[best_det_idx]
                self.tracks[track_id]['color'] = self.colors[color_idx] if color_idx < len(self.colors) else (255, 255, 0)
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)

        for det_idx, det_box in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                'bbox': det_box,
                'pred': predictions[det_idx],
                'conf': confidences[det_idx],
                'age': 0,
                'color': self.colors[predictions[det_idx]] if predictions[det_idx] < len(self.colors) else (255, 255, 0)
            }

        result = []
        for track_id, track in self.tracks.items():
            if track['age'] == 0:
                result.append({
                    'bbox': track['bbox'],
                    'pred': track['pred'],
                    'conf': track['conf'],
                    'color': track['color']
                })
        return result


class MaskDetector:
    def __init__(self, alpha=0.92):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
        self.bbox_smoother = BBoxSmoother(alpha=alpha, max_age=15, min_iou=0.2)
        self.labels = ['Mask OK', 'No Mask', 'Wrong Mask']
        self.fps = 0
        self.prev_time = None

    def _load_model(self):
        model_files = [
            'mask_classifier_final_complete.pth',
            'mask_classifier.pth',
            'mask_classifier_three_datasets_best.pth'
        ]
        model_path = None
        for f in model_files:
            if os.path.exists(f):
                model_path = f
                break
        if not model_path:
            available = [f for f in os.listdir('.') if f.endswith('.pth')]
            raise FileNotFoundError(f"Модель не найдена. Имеются: {available}")
        
        print(f" Загрузка модели: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        input_dim = checkpoint.get('input_dim', 119)
        model = MaskClassifier(input_dim=input_dim, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return frame

        raw_faces = self.face_detector.detect(frame)  
        detections, predictions, confidences = [], [], []

        for (x, y, w, h) in raw_faces:
            if w < 30 or h < 30:
                continue

            face_roi = frame[y:y+h, x:x+w]

            features = self.feature_extractor.extract(face_roi)

            expected_dim = self.model.fc1.in_features
            if features.shape[1] != expected_dim:
                if features.shape[1] < expected_dim:
                    features = np.pad(features, ((0, 0), (0, expected_dim - features.shape[1])), 'constant')
                else:
                    features = features[:, :expected_dim]

            with torch.no_grad():
                inp = torch.FloatTensor(features).to(self.device)
                out = self.model(inp)
                probs = F.softmax(out, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()

            detections.append((x, y, w, h))
            predictions.append(pred)
            confidences.append(conf)

        smoothed_tracks = self.bbox_smoother.update(detections, predictions, confidences)

        for track in smoothed_tracks:
            x, y, w, h = track['bbox']
            pred = track['pred']
            conf = track['conf']
            color = track['color']
            label = f"{self.labels[pred] if pred < len(self.labels) else 'Unknown'} ({conf:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        curr_time = time.time()
        if self.prev_time:
            dt = curr_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1 / dt)
        self.prev_time = curr_time

        fps_color = (0, 255, 0) if self.fps > 15 else (0, 165, 255) if self.fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        cv2.putText(frame, f"Faces: {len(smoothed_tracks)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def __call__(self, frame):
        return self.detect(frame.copy())

    def reset_tracks(self):
        self.bbox_smoother.tracks.clear()
        self.bbox_smoother.next_id = 0