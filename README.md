# Derin Pekiştirmeli Öğrenme Tabanlı Akıllı Kavşak Kontrol Sistemi

Bu proje, dört kollu bir kavşakta trafik ışıklarını **Derin Pekiştirmeli Öğrenme (DRL)** ile dinamik olarak kontrol eden bir simülasyon çalışmasıdır. Sistem; araç kuyruklarını, yaya bekleme süresini, acil araç önceliğini ve geçici şerit kapanmalarını birlikte değerlendirir.

## Proje Animasyonu

<p align="center">
  <img src="./traffic_control_demo_slow.gif" alt="DRL tabanlı akıllı kavşak simülasyonu" width="650">
</p>

## Eğitim Ödülü Grafiği

<p align="center">
  <img src="./training_curve.png" alt="DQN eğitim ödülü grafiği" width="850">
</p>

## Özellikler

- Dört kollu kavşak yapısı
- Her yönde düz, sol dönüş ve sağ dönüş şeritleri
- Dört yaya geçidi
- Yaya yoğunluğu ve yaya bekleme süresine göre yaya geçiş fazı
- Ambulans / itfaiye tespiti ve acil araç önceliği
- Dinamik yeşil ışık süresi seçimi: **8, 12 ve 16 saniye**
- Faz değişimlerinde sarı ışık güvenlik süresi
- Geçici şerit kapanması senaryosu
- Yoğun saat trafik talebi
- Double DQN, Replay Buffer, Target Network ve epsilon-greedy keşif stratejisi
- Eğitim ödülü grafiği ve GIF tabanlı simülasyon çıktısı

## Kullanılan Yöntem

Trafik ışığı kontrolü bir Markov Karar Süreci olarak modellenmiştir.

### Ajan

Trafik ışığı kontrol sistemi.

### Ortam

12 araç şeridi, 4 yaya geçidi, değişen trafik yoğunluğu, acil araç olayı ve geçici şerit kapanması bulunan dört kollu kavşak.

### Durum Uzayı

Ajan aşağıdaki bilgileri gözlemler:

- Şerit bazlı araç kuyrukları
- Şerit bazlı ortalama araç bekleme süreleri
- Yaya geçitlerindeki kişi sayıları
- Ortalama yaya bekleme süreleri
- Mevcut trafik ışığı fazı ve faz süresi
- Kapalı şerit bilgisi
- Acil aracın bulunduğu şerit ve bekleme süresi
- Normal, yoğun ve düşük trafik profili

### Aksiyon Uzayı

Ajan, 5 faz ve 3 yeşil ışık süresi arasından seçim yapar. Toplam aksiyon sayısı **15**’tir.

| Faz | Açıklama |
|---|---|
| 0 | Kuzey-Güney düz ve sağ dönüş |
| 1 | Doğu-Batı düz ve sağ dönüş |
| 2 | Kuzey-Güney sol dönüş |
| 3 | Doğu-Batı sol dönüş |
| 4 | Yaya geçiş fazı |

Her faz için 8, 12 veya 16 saniye yeşil ışık süresi seçilir.

## Güvenlik Katmanı

- Acil araç bekleme süresi belirlenen eşiği aşarsa ilgili şeride yeşil ışık sağlayan faz zorunlu olarak seçilir.
- Yaya bekleme süresi üst sınırı aşarsa yaya fazı zorunlu olarak açılır.

## Kurulum

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## Çalıştırma

```bash
python DRL_FinalProject.py
```

Google Colab üzerinde çalıştırmak için dosyaları yükledikten sonra:

```python
!pip install -r requirements.txt
!python DRL_FinalProject.py
```

## Oluşturulan Çıktılar

Program çalıştırıldığında `outputs_drl_traffic` klasöründe aşağıdaki dosyalar üretilir:

```text
outputs_drl_traffic/
├── traffic_dqn_model.pt
├── traffic_control_demo_slow.gif
└── training_curve.png
```

| Dosya | Açıklama |
|---|---|
| `traffic_dqn_model.pt` | Eğitilmiş Double DQN model ağırlıkları |
| `traffic_control_demo_slow.gif` | Trafik ışığı kararlarını gösteren animasyon |
| `training_curve.png` | Bölüm ödülleri ve 20 bölümlük hareketli ortalama |

## GitHub Dosya Yapısı

Tüm dosyalar aynı klasörde bulunur:

```text
repository/
├── DRL_FinalProject.py
├── README.md
├── requirements.txt
├── .gitignore
├── traffic_control_demo_slow.gif
└── training_curve.png
```

## Temel Hiperparametreler

| Parametre | Değer |
|---|---:|
| Eğitim bölümü | 500 |
| Maksimum simülasyon süresi | 360 saniye |
| Öğrenme oranı | 0.001 |
| Replay Buffer kapasitesi | 50.000 |
| Batch size | 128 |
| İndirim katsayısı | 0.995 |
| Başlangıç epsilon değeri | 1.0 |
| Minimum epsilon değeri | 0.05 |
| Yeşil ışık süreleri | 8 / 12 / 16 saniye |

## Geliştirme Fikirleri

- Sabit zamanlı trafik ışığı ile nicel performans karşılaştırması
- Çok ajanlı pekiştirmeli öğrenme ile birden fazla kavşak kontrolü
- SUMO veya gerçek trafik verisi entegrasyonu
- Hava koşulu, kaza ve yol çalışması senaryoları
- PPO, Dueling DQN veya farklı DRL algoritmalarının karşılaştırılması

## Lisans

Bu proje eğitim ve araştırma amaçlı hazırlanmıştır.
