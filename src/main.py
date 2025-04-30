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
    
    # Renk tanımlamaları
    colors = {
        'Mavi': {
            'range': ((100, 50, 50), (140, 255, 255)),
            'code': (255, 0, 0)
        },
        'Kahverengi': {
            'range': ((0, 50, 50), (30, 255, 255)),
            'code': (0, 75, 150)
        },
        'Yeşil': {
            'range': ((40, 50, 50), (80, 255, 255)),
            'code': (0, 255, 0)
        },
        'Diğer': {
            'range': ((0, 0, 0), (0, 0, 0)),  # Bu aralık kullanılmayacak
            'code': (128, 128, 128)
        }
    }
    
    # Göz bölgesi için maske oluştur
    h, w = eye_image.shape[:2]
    img_mask = np.zeros((h, w, 1), dtype=np.uint8)
    
    # Göz merkezi ve yarıçapı
    center = (w//2, h//2)
    radius = min(w, h) // 3  # Göz bölgesi için uygun yarıçap
    
    # Maske oluştur
    cv2.circle(img_mask, center, radius, (255, 255, 255), -1)
    
    # Renk dağılımını hesapla
    eye_class = np.zeros(len(colors), np.float32)
    
    def check_color(hsv, color_range):
        lower, upper = color_range
        return (hsv[0] >= lower[0] and hsv[0] <= upper[0] and
                hsv[1] >= lower[1] and hsv[1] <= upper[1] and
                hsv[2] >= lower[2] and hsv[2] <= upper[2])
    
    # Maskeli bölgedeki pikselleri analiz et
    for y in range(h):
        for x in range(w):
            if img_mask[y, x] > 0:
                hsv = hsv_eye[y, x]
                color_found = False
                for i, (color_name, color_info) in enumerate(colors.items()):
                    if color_name != 'Diğer' and check_color(hsv, color_info['range']):
                        eye_class[i] += 1
                        color_found = True
                        break
                if not color_found:
                    eye_class[-1] += 1  # Diğer kategorisine ekle
    
    # Renk yüzdelerini hesapla
    total_vote = eye_class.sum()
    if total_vote > 0:
        main_color_index = np.argmax(eye_class[:len(eye_class)-1])
        color_name = list(colors.keys())[main_color_index]
        return color_name, 0, colors[color_name]['code']
    else:
        return "Belirsiz", 0, colors['Diğer']['code']

def draw_dotted_circle(img, center, radius, color, thickness=1, dot_length=5, space_length=5):
    # Noktalı çember çizimi için yardımcı fonksiyon
    for angle in range(0, 360, dot_length + space_length):
        start_angle = angle
        end_angle = angle + dot_length
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

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
            
            # Gözler arası mesafeyi hesapla
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            eye_radius = int(eye_distance / 15)  # Dinamik göz yarıçapı
            
            # Sol göz bölgesi
            left_eye_region = frame[
                left_eye[1]-eye_radius:left_eye[1]+eye_radius,
                left_eye[0]-eye_radius:left_eye[0]+eye_radius
            ]
            
            # Sağ göz bölgesi
            right_eye_region = frame[
                right_eye[1]-eye_radius:right_eye[1]+eye_radius,
                right_eye[0]-eye_radius:right_eye[0]+eye_radius
            ]
            
            # Göz renklerini tespit et
            if left_eye_region.size > 0:
                left_color, _, left_color_code = get_eye_color(left_eye_region)
                # Arka plan için dikdörtgen çiz
                text = f"Sol: {left_color}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (10, 10), (20 + text_width, 30 + text_height), (0, 0, 0), -1)
                # Metni yaz
                cv2.putText(frame, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Noktalı çember çiz
                draw_dotted_circle(frame, left_eye, eye_radius, left_color_code, 1)
            
            if right_eye_region.size > 0:
                right_color, _, right_color_code = get_eye_color(right_eye_region)
                # Arka plan için dikdörtgen çiz
                text = f"Sag: {right_color}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (frame.shape[1] - 20 - text_width, 10), 
                            (frame.shape[1] - 10, 30 + text_height), (0, 0, 0), -1)
                # Metni yaz
                cv2.putText(frame, text, (frame.shape[1] - 15 - text_width, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Noktalı çember çiz
                draw_dotted_circle(frame, right_eye, eye_radius, right_color_code, 1)
        
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
        
        # Gözler arası mesafeyi hesapla
        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        eye_radius = int(eye_distance / 15)  # Dinamik göz yarıçapı
        
        # Sol göz bölgesi
        left_eye_region = frame[
            left_eye[1]-eye_radius:left_eye[1]+eye_radius,
            left_eye[0]-eye_radius:left_eye[0]+eye_radius
        ]
        
        # Sağ göz bölgesi
        right_eye_region = frame[
            right_eye[1]-eye_radius:right_eye[1]+eye_radius,
            right_eye[0]-eye_radius:right_eye[0]+eye_radius
        ]
        
        # Göz renklerini tespit et
        if left_eye_region.size > 0:
            left_color, _, left_color_code = get_eye_color(left_eye_region)
            # Arka plan için dikdörtgen çiz
            text = f"Sol: {left_color}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (10, 10), (20 + text_width, 30 + text_height), (0, 0, 0), -1)
            # Metni yaz
            cv2.putText(frame, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Noktalı çember çiz
            draw_dotted_circle(frame, left_eye, eye_radius, left_color_code, 1)
        
        if right_eye_region.size > 0:
            right_color, _, right_color_code = get_eye_color(right_eye_region)
            # Arka plan için dikdörtgen çiz
            text = f"Sağ: {right_color}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (frame.shape[1] - 20 - text_width, 10), 
                        (frame.shape[1] - 10, 30 + text_height), (0, 0, 0), -1)
            # Metni yaz
            cv2.putText(frame, text, (frame.shape[1] - 15 - text_width, 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Noktalı çember çiz
            draw_dotted_circle(frame, right_eye, eye_radius, right_color_code, 1)
    
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
    
    # Video hızını artırmak için bekleme süresini azalt
    wait_time = 1  # Normalde 1ms bekleme
    
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
            
            # Gözler arası mesafeyi hesapla
            eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            eye_radius = int(eye_distance / 15)  # Dinamik göz yarıçapı
            
            # Sol göz bölgesi
            left_eye_region = frame[
                left_eye[1]-eye_radius:left_eye[1]+eye_radius,
                left_eye[0]-eye_radius:left_eye[0]+eye_radius
            ]
            
            # Sağ göz bölgesi
            right_eye_region = frame[
                right_eye[1]-eye_radius:right_eye[1]+eye_radius,
                right_eye[0]-eye_radius:right_eye[0]+eye_radius
            ]
            
            # Göz renklerini tespit et
            if left_eye_region.size > 0:
                left_color, _, left_color_code = get_eye_color(left_eye_region)
                # Arka plan için dikdörtgen çiz
                text = f"Sol: {left_color}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (10, 10), (20 + text_width, 30 + text_height), (0, 0, 0), -1)
                # Metni yaz
                cv2.putText(frame, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Noktalı çember çiz
                draw_dotted_circle(frame, left_eye, eye_radius, left_color_code, 1)
            
            if right_eye_region.size > 0:
                right_color, _, right_color_code = get_eye_color(right_eye_region)
                # Arka plan için dikdörtgen çiz
                text = f"Sag: {right_color}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (frame.shape[1] - 20 - text_width, 10), 
                            (frame.shape[1] - 10, 30 + text_height), (0, 0, 0), -1)
                # Metni yaz
                cv2.putText(frame, text, (frame.shape[1] - 15 - text_width, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Noktalı çember çiz
                draw_dotted_circle(frame, right_eye, eye_radius, right_color_code, 1)
        
        # İşlenmiş kareyi kaydet
        out.write(frame)
        
        # Görüntüyü göster
        cv2.imshow('Göz Rengi Tespiti', frame)
        
        # Bekleme süresini azaltarak video hızını artır
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
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