# Göz Rengi Tespit Projesi

Bu proje, bilgisayarlı görü ve makine öğrenimi tekniklerini kullanarak görüntülerdeki göz renklerini tespit etmek ve analiz etmek için tasarlanmıştır.

## Özellikler

- Görüntülerde göz tespiti
- Renk analizi ve sınıflandırma
- Birden fazla görüntü formatı desteği

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/BaranAlt/guess_my_eye_color.git
cd guess_my_eye_color
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv eye_color_env
# Windows için
eye_color_env\Scripts\activate
# Unix veya MacOS için
source eye_color_env/bin/activate
```

3. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

⚠️ **Önemli Not**: Sanal ortam klasörünü (`eye_color_env`) Git'e eklemeyin. Bu klasör zaten `.gitignore` dosyasında belirtilmiştir.

## Proje Yapısı

```
eye_color_project/
├── src/           # Kaynak kodlar
├── assets/        # Proje varlıkları
└── samples/       # Örnek görüntüler
```

## Gereksinimler

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Pillow
- Matplotlib
