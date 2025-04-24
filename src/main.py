import cv2
import numpy as np
from mtcnn import MTCNN
import os

def get_eye_color(eye_image):
    # Görüntü ön işleme
    eye_image = cv2.GaussianBlur(eye_image, (5, 5), 0)  # Gürültüyü azalt
    
    # Kontrast artırma
    lab = cv2.cvtColor(eye_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    eye_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Görüntüyü HSV'ye dönüştür
    hsv_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)
    
    # Renk aralıkları
    color_ranges = {
    'Mavi': [
        ([90, 80, 40], [120, 255, 255]),  # Daha dar ve net mavi aralığı
        ([85, 40, 60], [120, 150, 200])   # Gri-mavi tonlar
    ],
    'Yeşil': [
        ([36, 80, 40], [85, 255, 200]),  # Doygun yeşil
        ([25, 40, 50], [35, 150, 180])   # Sarımsı yeşil
    ],
    'Kahverengi': [
        ([0, 50, 20], [20, 255, 120]),    # Koyu kahverengi
        ([5, 60, 30], [25, 200, 150])     # Açık kahverengi
    ],
    'Ela': [
        ([10, 40, 50], [25, 120, 200]),   # Altın-kahve
        ([25, 40, 50], [40, 120, 200])    # Yeşilimsi ela
    ],

}
    
    max_ratio = 0
    eye_color = "Belirsiz"
    max_color_confidence = {}
    
    # Her renk için maske oluştur ve oranını hesapla
    for color, ranges in color_ranges.items():
        color_ratio = 0
        
        # Her renk için birden fazla aralık olabilir
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Renk maskesi oluştur
            mask = cv2.inRange(hsv_eye, lower, upper)
            
            # Maskeyi temizle
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Bu aralık için oranı hesapla
            ratio = np.count_nonzero(mask) / mask.size
            color_ratio += ratio
        
        max_color_confidence[color] = color_ratio * 100
        
        # Mavi renk için ek ağırlık
        if color == 'Mavi':
            color_ratio *= 1.4  # Mavi renk tespitine %40 ek ağırlık
        
        # En yüksek orana sahip rengi bul
        if color_ratio > max_ratio:
            max_ratio = color_ratio
            eye_color = color
    
    return eye_color, max_ratio * 100

def test_camera():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
        for face in faces:
            # Yüz kutusunu çiz
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Göz konumlarını al
            keypoints = face['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Göz bölgelerini belirle
            eye_size = 20
            
            # Sol göz bölgesi
            left_eye_region = frame[
                left_eye[1]-eye_size:left_eye[1]+eye_size,
                left_eye[0]-eye_size:left_eye[0]+eye_size
            ]
            
            # Sağ göz bölgesi
            right_eye_region = frame[
                right_eye[1]-eye_size:right_eye[1]+eye_size,
                right_eye[0]-eye_size:right_eye[0]+eye_size
            ]
            
            # Göz renklerini tespit et
            if left_eye_region.size > 0:
                left_color, left_conf = get_eye_color(left_eye_region)
                cv2.putText(frame, f"Sol: {left_color} ({left_conf:.1f}%)",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if right_eye_region.size > 0:
                right_color, right_conf = get_eye_color(right_eye_region)
                cv2.putText(frame, f"Sağ: {right_color} ({right_conf:.1f}%)",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Göz bölgelerini çiz
            cv2.rectangle(frame, 
                        (left_eye[0]-eye_size, left_eye[1]-eye_size),
                        (left_eye[0]+eye_size, left_eye[1]+eye_size),
                        (255, 0, 0), 2)
            cv2.rectangle(frame,
                        (right_eye[0]-eye_size, right_eye[1]-eye_size),
                        (right_eye[0]+eye_size, right_eye[1]+eye_size),
                        (255, 0, 0), 2)
        
        cv2.imshow('Göz Rengi Tespiti', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    # Görüntüyü oku
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Hata: {image_path} dosyası okunamadı.")
        return
    
    detector = MTCNN()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    
    for face in faces:
        # Yüz kutusunu çiz
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
        # Göz konumlarını al
        keypoints = face['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Göz bölgelerini belirle
        eye_size = 20
        
        # Sol göz bölgesi
        left_eye_region = frame[
            left_eye[1]-eye_size:left_eye[1]+eye_size,
            left_eye[0]-eye_size:left_eye[0]+eye_size
        ]
        
        # Sağ göz bölgesi
        right_eye_region = frame[
            right_eye[1]-eye_size:right_eye[1]+eye_size,
            right_eye[0]-eye_size:right_eye[0]+eye_size
        ]
        
        # Göz renklerini tespit et
        if left_eye_region.size > 0:
            left_color, left_conf = get_eye_color(left_eye_region)
            cv2.putText(frame, f"Sol: {left_color} ({left_conf:.1f}%)",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if right_eye_region.size > 0:
            right_color, right_conf = get_eye_color(right_eye_region)
            cv2.putText(frame, f"Sağ: {right_color} ({right_conf:.1f}%)",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Göz bölgelerini çiz
        cv2.rectangle(frame, 
                    (left_eye[0]-eye_size, left_eye[1]-eye_size),
                    (left_eye[0]+eye_size, left_eye[1]+eye_size),
                    (255, 0, 0), 2)
        cv2.rectangle(frame,
                    (right_eye[0]-eye_size, right_eye[1]-eye_size),
                    (right_eye[0]+eye_size, right_eye[1]+eye_size),
                    (255, 0, 0), 2)
    
    # Sonuç görüntüsünü göster
    cv2.imshow('Göz Rengi Tespiti', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    
    # Video yazıcısını hazırla
    output_path = 'output_' + os.path.basename(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
        for face in faces:
            # Yüz kutusunu çiz
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
            # Göz konumlarını al
            keypoints = face['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            # Göz bölgelerini belirle
            eye_size = 20
            
            # Sol göz bölgesi
            left_eye_region = frame[
                left_eye[1]-eye_size:left_eye[1]+eye_size,
                left_eye[0]-eye_size:left_eye[0]+eye_size
            ]
            
            # Sağ göz bölgesi
            right_eye_region = frame[
                right_eye[1]-eye_size:right_eye[1]+eye_size,
                right_eye[0]-eye_size:right_eye[0]+eye_size
            ]
            
            # Göz renklerini tespit et
            if left_eye_region.size > 0:
                left_color, left_conf = get_eye_color(left_eye_region)
                cv2.putText(frame, f"Sol: {left_color} ({left_conf:.1f}%)",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if right_eye_region.size > 0:
                right_color, right_conf = get_eye_color(right_eye_region)
                cv2.putText(frame, f"Sağ: {right_color} ({right_conf:.1f}%)",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Göz bölgelerini çiz
            cv2.rectangle(frame, 
                        (left_eye[0]-eye_size, left_eye[1]-eye_size),
                        (left_eye[0]+eye_size, left_eye[1]+eye_size),
                        (255, 0, 0), 2)
            cv2.rectangle(frame,
                        (right_eye[0]-eye_size, right_eye[1]-eye_size),
                        (right_eye[0]+eye_size, right_eye[1]+eye_size),
                        (255, 0, 0), 2)
        
        # İşlenmiş kareyi kaydet
        out.write(frame)
        
        # Görüntüyü göster
        cv2.imshow('Göz Rengi Tespiti', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def list_sample_files(directory):
    """Belirtilen dizindeki dosyaları listeler"""
    files = []
    if os.path.exists(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

def select_file(files, file_type):
    """Kullanıcıya dosya seçtirme"""
    if not files:
        print(f"Uyarı: {file_type} klasöründe dosya bulunamadı!")
        return None
    
    print(f"\nMevcut {file_type} dosyaları:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input(f"\n{file_type} seçin (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            print("Geçersiz seçim!")
        except ValueError:
            print("Lütfen bir sayı girin!")

if __name__ == "__main__":
    print("Göz Rengi Tespit Programı")
    print("1. Kamera ile tespit")
    print("2. Fotoğraftan tespit")
    print("3. Videodan tespit")
    
    choice = input("Seçiminiz (1/2/3): ")
    
    if choice == "1":
        test_camera()
    elif choice == "2":
        # Fotoğrafları listele ve seç
        images_dir = "assets/images"
        image_files = list_sample_files(images_dir)
        selected_image = select_file(image_files, "fotoğraf")
        
        if selected_image:
            image_path = os.path.join(images_dir, selected_image)
            process_image(image_path)
        else:
            print("Lütfen assets/images klasörüne fotoğraf ekleyin!")
            
    elif choice == "3":
        # Videoları listele ve seç
        videos_dir = "assets/videos"
        video_files = list_sample_files(videos_dir)
        selected_video = select_file(video_files, "video")
        
        if selected_video:
            video_path = os.path.join(videos_dir, selected_video)
            process_video(video_path)
        else:
            print("Lütfen assets/videos klasörüne video ekleyin!")
    else:
        print("Geçersiz seçim!") 