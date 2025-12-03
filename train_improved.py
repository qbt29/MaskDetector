# augment_train_improved_masks_pytorch.py - УЛУЧШЕННАЯ классификация масок на PyTorch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def augment_image(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    face_roi = image[ymin:ymax, xmin:xmax]
    
    augmented_images = []
    
    augmented_images.append(face_roi)

    augmented_images.append(cv2.flip(face_roi, 1))
    
    for alpha in [0.6, 0.8, 1.2, 1.4]:
        bright = cv2.convertScaleAbs(face_roi, alpha=alpha, beta=0)
        augmented_images.append(bright)
    
    for beta in [-30, -15, 15, 30]:
        color_shift = cv2.convertScaleAbs(face_roi, alpha=1.0, beta=beta)
        augmented_images.append(color_shift)
    
    (h, w) = face_roi.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_roi, M, (w, h))
        augmented_images.append(rotated)
    
    noise = np.random.normal(0, 5, face_roi.shape).astype(np.uint8)
    noisy = cv2.add(face_roi, noise)
    augmented_images.append(noisy)
    
    for scale in [0.9, 1.1]:
        scaled = cv2.resize(face_roi, None, fx=scale, fy=scale)
        if scale < 1:
            padded = np.zeros_like(face_roi)
            h_s, w_s = scaled.shape[:2]
            padded[:h_s, :w_s] = scaled
            augmented_images.append(padded)
        else:
            cropped = scaled[:h, :w]
            augmented_images.append(cropped)
    
    return augmented_images

def extract_features_from_array(face_roi):
    try:
        face_roi_resized = cv2.resize(face_roi, (100, 100))
        
        h, w = face_roi_resized.shape[:2]
        
        top_roi = face_roi_resized[:h//3, :]
        middle_roi = face_roi_resized[h//3:2*h//3, :]
        bottom_roi = face_roi_resized[2*h//3:, :]
        
        hist_top_b = cv2.calcHist([top_roi], [0], None, [12], [0, 256])
        hist_top_g = cv2.calcHist([top_roi], [1], None, [12], [0, 256])
        hist_top_r = cv2.calcHist([top_roi], [2], None, [12], [0, 256])
        
        hist_bottom_b = cv2.calcHist([bottom_roi], [0], None, [12], [0, 256])
        hist_bottom_g = cv2.calcHist([bottom_roi], [1], None, [12], [0, 256])
        hist_bottom_r = cv2.calcHist([bottom_roi], [2], None, [12], [0, 256])
        
        avg_top = np.mean(top_roi, axis=(0, 1))
        avg_middle = np.mean(middle_roi, axis=(0, 1))
        avg_bottom = np.mean(bottom_roi, axis=(0, 1))
        
        diff_top_bottom = np.abs(avg_top - avg_bottom)
        diff_top_middle = np.abs(avg_top - avg_middle)
        diff_middle_bottom = np.abs(avg_middle - avg_bottom)
        
        hsv_top = cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV)
        hist_h_top = cv2.calcHist([hsv_top], [0], None, [10], [0, 180])
        hist_s_top = cv2.calcHist([hsv_top], [1], None, [10], [0, 256])
        
        gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)
        

        laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = np.var(sobelx) + np.var(sobely)
        
        brightness_top = np.mean(gray[:h//3, :])
        brightness_bottom = np.mean(gray[2*h//3:, :])
        brightness_contrast = np.abs(brightness_top - brightness_bottom)
        
        aspect_ratio = h / w if w > 0 else 1.0
        
        lab = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2LAB)
        avg_lab = np.mean(lab, axis=(0, 1))
        
        gray_std = np.std(gray)
        gray_mean = np.mean(gray)
        gray_skew = np.mean((gray - gray_mean)**3) / (gray_std**3) if gray_std > 0 else 0
        
        hist_top_b = cv2.normalize(hist_top_b, hist_top_b).flatten()
        hist_top_g = cv2.normalize(hist_top_g, hist_top_g).flatten()
        hist_top_r = cv2.normalize(hist_top_r, hist_top_r).flatten()
        hist_bottom_b = cv2.normalize(hist_bottom_b, hist_bottom_b).flatten()
        hist_bottom_g = cv2.normalize(hist_bottom_g, hist_bottom_g).flatten()
        hist_bottom_r = cv2.normalize(hist_bottom_r, hist_bottom_r).flatten()
        hist_h_top = cv2.normalize(hist_h_top, hist_h_top).flatten()
        hist_s_top = cv2.normalize(hist_s_top, hist_s_top).flatten()
        
        features = np.hstack([
            hist_top_b, hist_top_g, hist_top_r,      
            hist_bottom_b, hist_bottom_g, hist_bottom_r,
            
            hist_h_top, hist_s_top,                 
            
            avg_top, avg_middle, avg_bottom,         
            avg_lab,                               
            
            diff_top_bottom, diff_top_middle, diff_middle_bottom, 
            
            [laplacian_var, sobel_var, brightness_contrast, aspect_ratio, gray_std, gray_skew]
        ])
        # 119 признаков для дебага всех приложений
        if len(features) != 119:
            print(f"ВНИМАНИЕ: {len(features)} признаков вместо 119. Корректируем...")
            if len(features) < 119:
                features = np.pad(features, (0, 119 - len(features)), 'constant')
            else:
                features = features[:119]
        
        return features
    except Exception as e:
        print(f"Ошибка при извлечении признаков: {e}")
        return np.zeros(119)

class MaskClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(MaskClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        
        self.output = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        
        x = F.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01)
        x = self.output(x)
        return x

# Кастомный датасет
class MaskDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

def load_folder_dataset(dataset_path):
    """Загрузка второго датасета с папками"""
    features = []
    labels = []
    
    class_mapping = {
        'with_mask': 0,
        'without_mask': 1,
        'incorrect_mask': 2,
        'mask_incorrect': 2,
        'incorrect': 2
    }
    
    if not os.path.exists(dataset_path):
        print(f"Папка {dataset_path} не найдена!")
        return features, labels
    
    print(f"Загрузка датасета из папки: {dataset_path}")
    
    all_subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not all_subdirs:
        dataset_path_parent = os.path.dirname(dataset_path) if dataset_path.endswith('dataset') else dataset_path
        possible_class_dirs = ['with_mask', 'without_mask', 'incorrect_mask', 'mask_incorrect']
        
        for class_dir in possible_class_dirs:
            full_path = os.path.join(dataset_path_parent, class_dir)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                all_subdirs.append(full_path)
    
    if not all_subdirs:
        for class_name in class_mapping.keys():
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                all_subdirs.append(class_name)
    
    print(f"Найдено подпапок: {all_subdirs}")
    
    for folder_name in all_subdirs:
        folder_lower = folder_name.lower()
        class_id = None
        
        for class_name, class_num in class_mapping.items():
            if class_name in folder_lower:
                class_id = class_num
                break
        
        if class_id is None:
            continue
        
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            folder_path = folder_name
            if not os.path.exists(folder_path):
                continue
        
        print(f"Обработка папки: {folder_name} (класс: {class_id})")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        print(f"  Найдено {len(image_files)} изображений")
        
        if not image_files:
            continue
        
        for img_path in tqdm(image_files, desc=f"  {folder_name}", leave=False):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                
                crop_size = 0.8
                crop_h = int(h * crop_size)
                crop_w = int(w * crop_size)
                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                
                face_roi = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
                
                if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                    face_roi = image
                
                feat = extract_features_from_array(face_roi)
                if np.all(feat == 0):
                    continue
                
                features.append(feat)
                labels.append(class_id)
                
                if class_id in [1, 2]:
                    bbox = [0, 0, w, h]
                    augmented_faces = augment_image(image, bbox)
                    
                    for aug_face in augmented_faces:
                        if aug_face.size == 0:
                            continue
                        feat_aug = extract_features_from_array(aug_face)
                        if np.all(feat_aug == 0):
                            continue
                        features.append(feat_aug)
                        labels.append(class_id)
                else:
                    if np.random.random() < 0.3:
                        bbox = [0, 0, w, h]
                        augmented_faces = augment_image(image, bbox)
                        for aug_face in augmented_faces[:3]:
                            if aug_face.size == 0:
                                continue
                            feat_aug = extract_features_from_array(aug_face)
                            if np.all(feat_aug == 0):
                                continue
                            features.append(feat_aug)
                            labels.append(class_id)
                
            except Exception as e:
                print(f"Ошибка обработки {img_path}: {e}")
                continue
    
    return features, labels

def load_combined_datasets():
    features = []
    labels = []
    
    print("\n=== ЗАГРУЗКА КОМБИНИРОВАННЫХ ДАТАСЕТОВ ===")
    
    print("\n1. Загрузка ПЕРВОГО датасета (XML аннотации)...")
    
    image_files = glob.glob('images/*.png')
    if not image_files:
        image_files = glob.glob('images/*.jpg')
    
    print(f"   Найдено {len(image_files)} изображений")
    
    if image_files:
        dataset1_features, dataset1_labels = load_xml_dataset(image_files)
        if dataset1_features:
            features.extend(dataset1_features)
            labels.extend(dataset1_labels)
            print(f"   Загружено {len(dataset1_features)} образцов из первого датасета")
    
    print("\n2. Загрузка ВТОРОГО датасета (папки с классами)...")
    
    dataset_paths = [
        'dataset',
        'data/dataset',
        'dataset2',
        os.path.join(os.path.dirname(__file__), 'dataset')
    ]
    
    dataset2_found = False
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"   Найден датасет по пути: {dataset_path}")
            dataset2_features, dataset2_labels = load_folder_dataset(dataset_path)
            if dataset2_features:
                features.extend(dataset2_features)
                labels.extend(dataset2_labels)
                print(f"   Загружено {len(dataset2_features)} образцов из второго датасета")
                dataset2_found = True
                break
    
    if not dataset2_found:
        print("   Второй датасет не найден!")
    
    if not features:
        print("\nОШИБКА: Не загружено ни одного образца!")
        return np.array([]), np.array([])
    
    print(f"\n=== ЗАГРУЗКА ЗАВЕРШЕНА ===")
    print(f"Всего образцов: {len(features)}")
    
    if features:
        print(f"Размерность признаков: {len(features[0])}")
    
    return np.array(features), np.array(labels)

def load_xml_dataset(image_files):
    """Загрузка первого датасета с XML аннотациями"""
    features = []
    labels = []
    
    for img_path in tqdm(image_files, desc="Первый датасет"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = f"annotations/{base_name}.xml"
        
        if os.path.exists(xml_path):
            objects = parse_xml(xml_path)
            for obj in objects:
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    xmin, ymin, xmax, ymax = obj['bbox']
                    if xmin >= xmax or ymin >= ymax:
                        continue
                        
                    face_roi = image[ymin:ymax, xmin:xmax]
                    
                    if face_roi.size == 0:
                        continue
                        
                    feat = extract_features_from_array(face_roi)
                    if np.all(feat == 0):
                        continue
                        
                    features.append(feat)
                    labels.append(obj['class'])
                    
                    augmented_faces = augment_image(image, obj['bbox'])
                    
                    for aug_face in augmented_faces:
                        if aug_face.size == 0:
                            continue
                        feat_aug = extract_features_from_array(aug_face)
                        if np.all(feat_aug == 0):
                            continue
                        features.append(feat_aug)
                        labels.append(obj['class'])
                            
                except Exception as e:
                    print(f"Ошибка обработки {img_path}: {e}")
                    continue
    
    return features, labels

def train_model_with_augmentation():
    print("=== УЛУЧШЕННОЕ ОБУЧЕНИЕ КЛАССИФИКАТОРА МАСОК НА PyTorch ===")
    print("Загрузка КОМБИНИРОВАННЫХ данных с улучшенной аугментацией...")
    
    X, y = load_combined_datasets()
    
    if len(X) == 0:
        print("ОШИБКА: Данные не загружены!")
        print("Проверьте:")
        print("1. Наличие папки 'images' с изображениями и 'annotations' с XML файлами")
        print("2. Наличие папки 'dataset' с подпапками 'with_mask', 'without_mask', 'incorrect_mask'")
        return None
    
    print(f"\nЗагружено {len(X)} образцов (с аугментацией)")
    
    class_counts = np.unique(y, return_counts=True)
    print(f"\nРаспределение классов после аугментации:")
    total_samples = len(y)
    for cls, count in zip(class_counts[0], class_counts[1]):
        class_names = ['with_mask', 'without_mask', 'mask_incorrect']
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {class_names[cls]}: {count} образцов ({percentage:.1f}%)")
    
    print("\nБалансировка классов...")
    max_class_count = max(class_counts[1]) if len(class_counts[1]) > 0 else 0
    
    balanced_features = []
    balanced_labels = []
    
    for cls in range(3):
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)
        
        if cls_count == 0:
            print(f"  Предупреждение: класс {cls} не имеет образцов!")
            continue

        balanced_features.extend(X[cls_indices])
        balanced_labels.extend([cls] * cls_count)
        
        if cls_count < max_class_count:
            shortage = max_class_count - cls_count
            print(f"  Класс {cls}: добавление {shortage} аугментированных образцов")
            
            for _ in range(shortage):
                random_idx = np.random.choice(cls_indices)
                sample = X[random_idx].copy()
                noise = np.random.normal(0, 0.01, sample.shape)
                augmented = sample + noise
                balanced_features.append(augmented)
                balanced_labels.append(cls)
    
    X = np.array(balanced_features)
    y = np.array(balanced_labels)
    
    print(f"\nПосле балансировки: {len(X)} образцов")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"\nРазмеры данных:")
    print(f"  Обучающая выборка: {X_train.shape} ({len(y_train)} образцов)")
    print(f"  Валидационная выборка: {X_val.shape} ({len(y_val)} образцов)")
    print(f"  Тестовая выборка: {X_test.shape} ({len(y_test)} образцов)")
    print(f"  Размерность признаков: {X_train.shape[1]}")
    
    train_dataset = MaskDataset(X_train, y_train)
    val_dataset = MaskDataset(X_val, y_val)
    test_dataset = MaskDataset(X_test, y_test)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    input_dim = X_train.shape[1]
    model = MaskClassifier(input_dim, num_classes=3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуемое устройство: {device}")
    model.to(device)
    
    class_counts = np.bincount(y_train)
    print(f"\nРаспределение в тренировочных данных:")
    for i, count in enumerate(class_counts):
        class_name = ['with_mask', 'without_mask', 'mask_incorrect'][i]
        percentage = (count / len(y_train) * 100) if len(y_train) > 0 else 0
        print(f"  {class_name}: {count} образцов ({percentage:.1f}%)")
    
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    print(f"Веса классов: {class_weights}")
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    
    num_epochs = 100
    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience = 20
    patience_counter = 0
    
    print("\n" + "="*60)
    print(f"НАЧИНАЕМ ОБУЧЕНИЕ ({num_epochs} эпох, batch_size={batch_size})")
    print("="*60)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs}", leave=False)
        for batch_features, batch_labels in train_pbar:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * train_correct / train_total
            })
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy,
                'input_dim': input_dim,
                'class_names': ['with_mask', 'without_mask', 'mask_incorrect'],
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'feature_dimension': 119, 
                'num_features': 119
            }, 'mask_classifier.pth')
            patience_counter = 0
            print(f"✓ Новая лучшая модель сохранена (val_acc: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
        
        print(f"\nЭпоха {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"  Best Val Acc: {best_val_accuracy:.2f}% | LR: {current_lr:.6f}")
        print(f"  Паттерн: {patience_counter}/{patience}")
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'input_dim': input_dim
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  Чекпоинт сохранен: checkpoint_epoch_{epoch+1}.pth")
        
        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch+1}")
            break
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    try:
        checkpoint = torch.load('mask_classifier.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружена лучшая модель с эпохи {checkpoint['epoch']+1}")
        print(f"Лучшая val_accuracy: {checkpoint['val_accuracy']:.2f}%")
    except Exception as e:
        print(f"Ошибка загрузки лучшей модели: {e}")
        print("Используется последняя модель")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_test_preds = []
    all_test_labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Тестирование")
        for batch_features, batch_labels in test_pbar:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(batch_labels.cpu().numpy())
            
            test_pbar.set_postfix({'acc': 100 * test_correct / test_total})
    
    test_accuracy = 100 * test_correct / test_total
    print(f"\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"Точность на тестовой выборке: {test_accuracy:.2f}%")
    
    print("\nДетальный отчет классификации:")
    print(classification_report(
        all_test_labels, all_test_preds,
        target_names=['with_mask', 'without_mask', 'mask_incorrect'],
        zero_division=0,
        digits=4
    ))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print("\nМатрица ошибок:")
    print(cm)
    

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'class_names': ['with_mask', 'without_mask', 'mask_incorrect'],
        'test_accuracy': test_accuracy,
        'feature_extractor_info': 'Используйте extract_features_from_array() для извлечения признаков',
        'dataset_info': 'Обучена на комбинированных датасетах (XML + папки)',
        'training_params': {
            'batch_size': batch_size,
            'num_epochs_trained': epoch + 1,
            'best_val_accuracy': best_val_accuracy,
            'num_features': input_dim,
            'feature_dimension': 119  
        }
    }, 'mask_classifier_final_complete.pth')
    
    print("\n" + "="*60)
    print("МОДЕЛИ СОХРАНЕНЫ:")
    print("="*60)
    print("1. 'mask_classifier.pth' - лучшая модель")
    print("2. 'mask_classifier_final_complete.pth' - финальная модель")
    print(f"3. Размерность признаков: {input_dim}")
    
    unique, counts = np.unique(all_test_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        correct = np.sum((np.array(all_test_labels) == cls) & (np.array(all_test_preds) == cls))
        accuracy = correct / count * 100 if count > 0 else 0
        class_name = ['with_mask', 'without_mask', 'mask_incorrect'][cls]
        print(f"  {class_name}: {count} образцов, точность: {accuracy:.1f}%")
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_final.png', dpi=300)
        print("\n✓ Графики обучения сохранены в 'training_history_final.png'")
        plt.show()
    except Exception as e:
        print(f"\nНе удалось построить графики: {e}")
    
    return model

def predict_mask(features_array, model_path='mask_classifier_final_complete.pth'):
    """Предсказание класса маски для извлеченных признаков"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = MaskClassifier(checkpoint['input_dim'], num_classes=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    features_tensor = torch.FloatTensor(features_array).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return {
        'class': predicted_class,
        'class_name': checkpoint['class_names'][predicted_class],
        'probabilities': probabilities.numpy().flatten(),
        'confidence': float(probabilities.max().item())
    }

def quick_test():
    """Быстрый тест модели"""
    print("\n" + "="*50)
    print("БЫСТРЫЙ ТЕСТ МОДЕЛИ")
    print("="*50)
    
    test_features = np.random.randn(10, 119)
    test_labels = np.random.choice([0, 1, 2], 10)
    
    model = MaskClassifier(119, num_classes=3)
    
    test_tensor = torch.FloatTensor(test_features[:3])
    output = model(test_tensor)
    print(f"Размер выхода модели: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nОбщее количество параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    print("\n✓ Модель работает корректно!")

if __name__ == "__main__":
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступно: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    quick_test()
    
    print("\n" + "="*60)
    print("ЗАПУСК ОБУЧЕНИЯ ПОЛНОЙ МОДЕЛИ")
    print("="*60)
    model = train_model_with_augmentation()