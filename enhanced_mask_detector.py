# enhanced_mask_detector.py - ГИБРИДНЫЙ ДЕТЕКТОР МАСОК (Geometry + ML only)
# Улучшения:
#   • MediaPipe Face Detection вместо Haar
#   • Анализ ТОЛЬКО нижней трети лица (как при обучении)
#   • Вход в модель строго выровнен под трейн (100×100, нижняя треть → паддинг)
#   • Убрана цветовая логика — только ML

import cv2
import numpy as np
import joblib
import sys
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

sys.path.append(os.path.dirname(__file__))

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh  # ← ЭТА СТРОКА ОБЯЗАТЕЛЬНА

# ===== MEDIAPIPE FACE DETECTION (lite, fast) =====
mp_face_detection = mp.solutions.face_detection

def calculate_head_coordinates(frame, face_center):
    h, w = frame.shape[:2]
    face_x = int(face_center[0] * w)
    face_y = int(face_center[1] * h)
    x1 = max(0, face_x - 120)
    y1 = max(0, face_y - 150)  # чуть больше по вертикали
    x2 = min(w, face_x + 120)
    y2 = min(h, face_y + 100)
    return face_x, face_y, x1, y1, x2, y2

def draw_skeleton(frame, landmarks, mp_drawing):
    connections = create_custom_connections()
    drawing_specs = (
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    draw_custom_pose_landmarks(frame, landmarks, connections, drawing_specs)

def detect_faces_mp(roi, face_detector):
    """Точная детекция лиц в ROI через MediaPipe"""
    if roi.size == 0:
        return []
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_roi)
    faces = []
    if results.detections:
        h_roi, w_roi = roi.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            if bbox:
                x = int(bbox.xmin * w_roi)
                y = int(bbox.ymin * h_roi)
                w = int(bbox.width * w_roi)
                h = int(bbox.height * h_roi)
                if w > 30 and h > 30:
                    faces.append((x, y, w, h))
    return faces

# 🔑 ТОЖЕ САМОЕ, ЧТО В ТРЕЙНЕ! (обязательно!)
def extract_mask_features(face_roi):
    """
    Извлечение признаков ТОЧНО КАК ПРИ ОБУЧЕНИИ.
    На входе: изображение ~100×100 (предполагается, что это лицо или его нижняя часть).
    """
    face_roi_resized = cv2.resize(face_roi, (100, 100))
    h, w = face_roi_resized.shape[:2]

    # === КЛЮЧ: ТО ЖЕ РАЗБИЕНИЕ, ЧТО В augment_train_improved_masks.py ===
    top_roi = face_roi_resized[:h//3, :]          # верхняя треть — маска (в трейне это top)
    middle_roi = face_roi_resized[h//3:2*h//3, :]
    bottom_roi = face_roi_resized[2*h//3:, :]     # нижняя треть — подбородок

    # Гистограммы
    def get_hist(region, channels):
        hist = []
        for ch in channels:
            hch = cv2.calcHist([region], [ch], None, [12], [0, 256])
            hch = cv2.normalize(hch, hch).flatten()
            hist.append(hch)
        return np.hstack(hist)

    hist_top = get_hist(top_roi, [0, 1, 2])        # BGR
    hist_bottom = get_hist(bottom_roi, [0, 1, 2])  # BGR

    # HSV для top
    hsv_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_top], [0], None, [10], [0, 180])
    hist_s = cv2.calcHist([hsv_top], [1], None, [10], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()

    # Средние цвета
    avg_top = np.mean(top_roi, axis=(0, 1))
    avg_middle = np.mean(middle_roi, axis=(0, 1))
    avg_bottom = np.mean(bottom_roi, axis=(0, 1))

    # Разницы
    diff_tb = np.abs(avg_top - avg_bottom)
    diff_tm = np.abs(avg_top - avg_middle)
    diff_mb = np.abs(avg_middle - avg_bottom)

    # Текстура
    gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
    lap_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = np.var(sobelx) + np.var(sobely)

    # Яркость контраст
    brightness_top = np.mean(gray[:h//3, :])
    brightness_bottom = np.mean(gray[2*h//3:, :])
    brightness_contrast = abs(brightness_top - brightness_bottom)

    # Сборка
    features = np.hstack([
        hist_top, hist_bottom,
        hist_h, hist_s,
        avg_top, avg_middle, avg_bottom,
        diff_tb, diff_tm, diff_mb,
        [lap_var, sobel_var, brightness_contrast]
    ])
    return features.reshape(1, -1)

def prepare_lower_third_for_model(head_roi):
    """
    Подготавливает нижнюю треть головы/лица для подачи в модель.
    Возвращает (100, 100, 3) изображение, совместимое с трейном.
    """
    if head_roi.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    h, w = head_roi.shape[:2]
    # Берём НИЖНЮЮ ТРЕТЬ (где рот и подбородок — зона маски!)
    lower_third = head_roi[2 * h // 3:, :]  # ~33% снизу
    
    if lower_third.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Ресайз до высоты ~33, ширина пропорциональна
    target_h = 33
    scale = target_h / lower_third.shape[0]
    target_w = max(1, int(lower_third.shape[1] * scale))
    resized = cv2.resize(lower_third, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Паддинг до 100×100 (маска внизу, как на трейновых ROI)
    padded = np.zeros((100, 100, 3), dtype=np.uint8)
    y_offset = 100 - target_h  # прижимаем вниз
    x_offset = max(0, (100 - target_w) // 2)
    x_end = min(100, x_offset + target_w)
    padded[y_offset:, x_offset:x_end] = resized[:, :x_end - x_offset]
    
    return padded

def draw_head_analysis(frame, x1_head, y1_head, x2_head, y2_head, head_roi, face_detector, mask_model):
    """
    ЧИСТЫЙ ML-подход:
      1. Найти лица в head ROI (MediaPipe почти всегда находит)
      2. Для каждого лица — вырезать → 100×100 → predict
      3. Никаких fallback'ов, никаких "если нет лица"
    """
    faces = detect_faces_mp(head_roi, face_detector)

    if len(faces) == 0:
        # Редкий случай — рисуем head bbox как fallback
        cv2.rectangle(frame, (x1_head, y1_head), (x2_head, y2_head), (128, 128, 128), 2)
        cv2.putText(frame, "No face", (x1_head, y1_head - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return

    # Обрабатываем каждое найденное лицо
    for (fx, fy, fw, fh) in faces:
        # Глобальные координаты
        x1 = x1_head + fx
        y1 = y1_head + fy
        x2 = x1 + fw
        y2 = y1 + fh

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        try:
            #  КЛЮЧЕВОЙ ПОТОК:
            face_100 = cv2.resize(face_roi, (100, 100))      # как в трейне
            features = extract_mask_features(face_100)       # ваша функция — точная копия трейна
            probas = mask_model.predict_proba(features)[0]
            pred = int(np.argmax(probas))
            conf = float(probas[pred])

            # Классы
            labels = ['Mask OK', 'No Mask', 'Wrong Mask']
            colors = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]

            label = labels[pred]
            color = colors[pred]

            # Отрисовка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            txt_color = (255, 255, 255) if np.mean(color) < 128 else (0, 0, 0)
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

        except Exception as e:
            # Отладка ошибки (временно)
            print(f" ML error: {e}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Error", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_face_center(frame, face_x, face_y):
    cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)

def main():
    print("===  MULTI-PERSON MASK DETECTOR (Face Mesh + ML) ===")
    print(" Нативная поддержка нескольких лиц, без дублей")
    print("Press ESC to exit\n")

    # Загрузка модели — как раньше
    mask_model = None
    model_name = None
    for model_path in [
        'mask_classifier_super_improved.pkl',
        'mask_classifier_fixed.pkl',
        'mask_classifier_augmented.pkl'
    ]:
        if os.path.exists(model_path):
            try:
                mask_model = joblib.load(model_path)
                model_name = model_path
                print(f" ML-модель загружена: '{model_name}'")
                break
            except Exception as e:
                print(f" Ошибка загрузки '{model_path}': {e}")
    if mask_model is None:
        print(" ML-модель не найдена.")
        return

    #  ИСПОЛЬЗУЕМ FaceMesh с поддержкой нескольких лиц
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,                      # ← сколько лиц максимум
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print(" Face Mesh initialized (multi-face, no duplicates)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Камера недоступна")
        return

    print(" Запуск детекции...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #  ОБНАРУЖЕНИЕ ВСЕХ ЛИЦ ЗА ОДИН ПРОХОД
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            all_faces = []
            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                for flm in results.multi_face_landmarks:
                    # Получаем bbox напрямую из 468 точек
                    xs = [lm.x for lm in flm.landmark]
                    ys = [lm.y for lm in flm.landmark]
                    x1 = int(min(xs) * w)
                    y1 = int(min(ys) * h)
                    x2 = int(max(xs) * w)
                    y2 = int(max(ys) * h)

                    # Добавляем padding (~10%)
                    pad_w = int(0.1 * (x2 - x1))
                    pad_h = int(0.1 * (y2 - y1))
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)

                    all_faces.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, w, h)

            # Обработка каждого лица — как раньше
            for (x, y, w_box, h_box) in all_faces:
                face_roi = frame[y:y+h_box, x:x+w_box]
                if face_roi.size == 0:
                    continue

                try:
                    face_100 = cv2.resize(face_roi, (100, 100))
                    features = extract_mask_features(face_100)
                    probas = mask_model.predict_proba(features)[0]
                    pred = int(np.argmax(probas))
                    conf = float(probas[pred])

                    labels = ['Mask OK', 'No Mask', 'Wrong Mask']
                    colors = [(0, 255, 0), (0, 0, 255), (0, 165, 255)]

                    label = labels[pred]
                    color = colors[pred]

                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    text = f"{label} ({conf:.2f})"
                    txt_color = (255, 255, 255) if np.mean(color) < 128 else (0, 0, 0)
                    cv2.putText(frame, text, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

                except Exception as e:
                    print(f" ML error on face at ({x},{y}): {e}")
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)
                    cv2.putText(frame, "Error", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Информация
            cv2.putText(frame, f"Faces: {len(all_faces)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Multi-Person Mask Detector', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print("\n Детекция завершена")
        
if __name__ == "__main__":
    main()