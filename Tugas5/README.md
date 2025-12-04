# Real-time rPPG Heart Rate Monitor using MediaPipe + OpenCV

Proyek ini adalah aplikasi real-time heart rate (detak jantung) menggunakan metode rPPG (remote Photoplethysmography) yang diekstraksi dari kamera webcam.
Sistem mendeteksi wajah menggunakan MediaPipe, mengambil sinyal RGB dari area dahi/pipi, kemudian memproses sinyal tersebut untuk mengekstrak detak jantung dengan filtering, FFT, dan peak detection.

Akurasi meningkat berkat:

- Detrending sinyal
- Bandpass filter (0.67–4 Hz)
- FFT frequency analysis
- Peak detection dengan outlier removal
- Median + exponential smoothing
- FPS compensation untuk stabilitas

---

## Fitur Utama

✔Real-time face detection (MediaPipe)
✔ ROI tracking pada dahi/pipi
✔ Ekstraksi sinyal rPPG berbasis RGB
✔ POS algorithm (Plane Orthogonal to Skin)
✔ Detrending + bandpass filtering
✔ BPM estimation dengan:

- FFT (Fast Fourier Transform)
- Peak detection
  ✔ Automatic FPS calibration
  ✔ Stabil & robust smoothing
  ✔ Real-time GUI overlay menggunakan OpenCV

---

## Dependencies dan Cara Menjalankan

Pastikan sudah menginstall library berikut :

```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

Jalankan script:

```bash
python 122140048.py
```

## Cara Menggunakan

- Pastikan wajah berada di tengah webcam.
- Jangan banyak bergerak.
- Pencahayaan harus cukup (tidak terlalu gelap/red).
- Tunggu 5 detik hingga buffer terkumpul (progress bar).
- BPM akan muncul secara otomatis.
- Klik q jika ingin leave/meninggalkan meet