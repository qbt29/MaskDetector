# augment_train_improved_masks.py - УЛУЧШЕННАЯ классификация масок
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import glob
import xml.etree.ElementTree as ET

# УЛУЧШЕННАЯ аугментация для масок
def augment_image(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    face_roi = image[ymin:ymax, xmin:xmax]
    
    augmented_images = []
    
    # Оригинал
    augmented_images.append(face_roi)
    
    # Горизонтальное отражение
    augmented_images.append(cv2.flip(face_roi, 1))
    
    # Разные уровни яркости (важно для масок)
    for alpha in [0.6, 0.8, 1.2, 1.4]:
        bright = cv2.convertScaleAbs(face_roi, alpha=alpha, beta=0)
        augmented_images.append(bright)
    
    # Цветовые искажения (имитируют разные маски)
    for beta in [-30, -15, 15, 30]:  # Сдвиг цвета
        color_shift = cv2.convertScaleAbs(face_roi, alpha=1.0, beta=beta)
        augmented_images.append(color_shift)
    
    return augmented_images

# СУПЕР-УЛУЧШЕННОЕ извлечение признаков для масок
def extract_features_from_array(face_roi):
    face_roi_resized = cv2.resize(face_roi, (100, 100))
    
    # 1. Цветовые гистограммы РАЗНЫХ ОБЛАСТЕЙ (ключевое для масок)
    h, w = face_roi_resized.shape[:2]
    
    # Верхняя треть (где маска)
    top_roi = face_roi_resized[:h//3, :]
    # Средняя треть (нос/щеки)
    middle_roi = face_roi_resized[h//3:2*h//3, :]
    # Нижняя треть (рот/подбородок)
    bottom_roi = face_roi_resized[2*h//3:, :]
    
    # Гистограммы для каждой области
    hist_top_b = cv2.calcHist([top_roi], [0], None, [12], [0, 256])
    hist_top_g = cv2.calcHist([top_roi], [1], None, [12], [0, 256])
    hist_top_r = cv2.calcHist([top_roi], [2], None, [12], [0, 256])
    
    hist_bottom_b = cv2.calcHist([bottom_roi], [0], None, [12], [0, 256])
    hist_bottom_g = cv2.calcHist([bottom_roi], [1], None, [12], [0, 256])
    hist_bottom_r = cv2.calcHist([bottom_roi], [2], None, [12], [0, 256])
    
    # 2. СРЕДНИЕ ЦВЕТА и КОНТРАСТ между областями
    avg_top = np.mean(top_roi, axis=(0, 1))
    avg_middle = np.mean(middle_roi, axis=(0, 1))
    avg_bottom = np.mean(bottom_roi, axis=(0, 1))
    
    # Разницы цветов между областями (очень важно!)
    diff_top_bottom = np.abs(avg_top - avg_bottom)
    diff_top_middle = np.abs(avg_top - avg_middle)
    diff_middle_bottom = np.abs(avg_middle - avg_bottom)
    
    # 3. HSV признаки для верхней области (где маска)
    hsv_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
    hist_h_top = cv2.calcHist([hsv_top], [0], None, [10], [0, 180])
    hist_s_top = cv2.calcHist([hsv_top], [1], None, [10], [0, 256])
    
    # 4. ТЕКСТУРА - маски часто имеют другую текстуру
    gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
    
    # Разные типы текстур
    laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_var = np.var(sobelx) + np.var(sobely)
    
    # 5. ЯРКОСТЬ и КОНТРАСТ разных областей
    brightness_top = np.mean(gray[:h//3, :])
    brightness_bottom = np.mean(gray[2*h//3:, :])
    brightness_contrast = np.abs(brightness_top - brightness_bottom)
    
    # Нормализация всех гистограмм
    hist_top_b = cv2.normalize(hist_top_b, hist_top_b).flatten()
    hist_top_g = cv2.normalize(hist_top_g, hist_top_g).flatten()
    hist_top_r = cv2.normalize(hist_top_r, hist_top_r).flatten()
    hist_bottom_b = cv2.normalize(hist_bottom_b, hist_bottom_b).flatten()
    hist_bottom_g = cv2.normalize(hist_bottom_g, hist_bottom_g).flatten()
    hist_bottom_r = cv2.normalize(hist_bottom_r, hist_bottom_r).flatten()
    hist_h_top = cv2.normalize(hist_h_top, hist_h_top).flatten()
    hist_s_top = cv2.normalize(hist_s_top, hist_s_top).flatten()
    
    # ОБЪЕДИНЯЕМ ВСЕ УЛУЧШЕННЫЕ ПРИЗНАКИ
    features = np.hstack([
        # Цветовые гистограммы областей
        hist_top_b, hist_top_g, hist_top_r,
        hist_bottom_b, hist_bottom_g, hist_bottom_r,
        hist_h_top, hist_s_top,
        
        # Средние цвета
        avg_top, avg_middle, avg_bottom,
        
        # Разницы цветов
        diff_top_bottom, diff_top_middle, diff_middle_bottom,
        
        # Текстура и яркость
        [laplacian_var, sobel_var, brightness_contrast]
    ])
    
    return features

# Загрузка датасета с УСИЛЕННОЙ аугментацией
def load_dataset_with_augmentation():
    features = []
    labels = []
    
    image_files = glob.glob('images/*.png')
    print(f"Найдено {len(image_files)} изображений")
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = f"annotations/{base_name}.xml"
        
        if os.path.exists(xml_path):
            objects = parse_xml(xml_path)
            for obj in objects:
                try:
                    # Основной образец
                    image = cv2.imread(img_path)
                    xmin, ymin, xmax, ymax = obj['bbox']
                    face_roi = image[ymin:ymax, xmin:xmax]
                    
                    feat = extract_features_from_array(face_roi)
                    features.append(feat)
                    labels.append(obj['class'])
                    
                    # УСИЛЕННАЯ аугментация для ВСЕХ классов
                    if obj['class'] in [1, 2]:  # without_mask и mask_incorrect
                        if face_roi.size > 0:
                            augmented_faces = augment_image(image, obj['bbox'])
                            
                            for aug_face in augmented_faces:
                                feat_aug = extract_features_from_array(aug_face)
                                features.append(feat_aug)
                                labels.append(obj['class'])
                    else:  # with_mask - тоже немного аугментируем
                        if face_roi.size > 0 and np.random.random() < 0.3:  # 30% аугментации
                            augmented_faces = augment_image(image, obj['bbox'])
                            for aug_face in augmented_faces[:2]:  # только 2 аугментации
                                feat_aug = extract_features_from_array(aug_face)
                                features.append(feat_aug)
                                labels.append(obj['class'])
                            
                except Exception as e:
                    print(f"Ошибка обработки {img_path}: {e}")
    
    return np.array(features), np.array(labels)

# Остальные функции остаются прежними
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    class_mapping = {'with_mask': 0, 'without_mask': 1, 'mask_weared_incorrect': 2}
    objects = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'class': class_mapping[class_name],
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return objects

def extract_features(image_path, bbox):
    image = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = bbox
    face_roi = image[ymin:ymax, xmin:xmax]
    return extract_features_from_array(face_roi)

def train_model_with_augmentation():
    print("=== УЛУЧШЕННОЕ ОБУЧЕНИЕ КЛАССИФИКАТОРА МАСОК ===")
    print("Загрузка данных с улучшенной аугментацией...")
    X, y = load_dataset_with_augmentation()
    
    print(f"Загружено {len(X)} образцов (с аугментацией)")
    class_counts = np.unique(y, return_counts=True)
    print(f"Распределение классов после аугментации: {class_counts}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Размерность признаков: {X_train.shape[1]}")
    
    # СУПЕР-УЛУЧШЕННАЯ МОДЕЛЬ для масок
    model = RandomForestClassifier(
        n_estimators=300,  # Еще больше деревьев
        max_depth=30,      # Глубже
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        max_features='sqrt'  # Лучшее обобщение
    )
    
    print("Обучение улучшенной модели для масок...")
    model.fit(X_train, y_train)
    
    # Оценка
    accuracy = model.score(X_test, y_test)
    print(f"Точность: {accuracy:.3f}")
    
    y_pred = model.predict(X_test)
    print("\nДетальный отчет классификации:")
    print(classification_report(y_test, y_pred, 
                              target_names=['with_mask', 'without_mask', 'mask_incorrect'],
                              zero_division=0))
    
    joblib.dump(model, 'mask_classifier_super_improved.pkl')
    print("СУПЕР-улучшенная модель сохранена как 'mask_classifier_super_improved.pkl'")
    
    return model

if __name__ == "__main__":
    train_model_with_augmentation()