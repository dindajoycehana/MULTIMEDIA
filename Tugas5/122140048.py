# Import dependencies
import cv2 
import numpy as np
import mediapipe as mp
import scipy.signal as signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import deque
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Fungsi POS (Plane Orthogonal to Skin)
def POS(signal_data, **kwargs):
    """
    Implementasi algoritma POS untuk ekstraksi sinyal rPPG
    """
    eps = 10**-9
    X = signal_data
    e, c, f = X.shape  # estimator, channel (RGB), frame
    w = int(1.6 * kwargs['fps'])  # Window length in frames

    # P: Matriks transformasi warna dari RGB ke proyeksi POS
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # H: Matriks keluaran
    H = np.zeros((e, f))
    
    for n in range(w, f):
        m = n - w + 1

        # Normalisasi sinyal dari pengaruh cahaya
        Cn = X[:, :, m:(n+1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(Cn, M)

        # Proyeksikan sinyal RGB ke dalam domain POS
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)

        # Tuning sinyal
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)

        # Overlap-add
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

# Fungsi untuk detrending menggunakan sliding average
def detrend_signal(signal_data, window_size=30):
    """
    Menghilangkan trend dari sinyal menggunakan sliding average
    """
    if len(signal_data) < window_size:
        return signal_data - np.mean(signal_data)
    
    # Moving average dengan padding
    moving_avg = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
    # Perbaiki edge effects dengan mirroring
    detrended = signal_data - moving_avg
    return detrended / (np.std(detrended) + 1e-9)

# Fungsi untuk bandpass filter
def bandpass_filter(signal_data, fps, lowcut=0.67, highcut=4.0):
    """
    Bandpass filter untuk frekuensi jantung (40-240 BPM)
    lowcut = 0.67 Hz (40 BPM)
    highcut = 4.0 Hz (240 BPM)
    """
    if len(signal_data) < 20:
        return signal_data
    
    nyquist = fps / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Pastikan nilai dalam range [0, 1)
    low = max(0.01, min(low, 0.99))
    high = max(0.01, min(high, 0.99))
    
    if low >= high:
        return signal_data
    
    # Gunakan order lebih tinggi untuk filter yang lebih sharp
    try:
        b, a = signal.butter(5, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    except:
        # Fallback jika filter gagal
        return signal_data

# Fungsi untuk estimasi BPM menggunakan FFT
def estimate_bpm_fft(signal_data, fps):
    """
    Estimasi BPM menggunakan Fast Fourier Transform dengan smoothing
    """
    if len(signal_data) < fps * 2:  # Minimal 2 detik data
        return 0
    
    # Terapkan Hamming window untuk mengurangi spectral leakage
    windowed_signal = signal_data * np.hamming(len(signal_data))
    
    # FFT
    fft_data = np.fft.fft(windowed_signal)
    fft_freq = np.fft.fftfreq(len(signal_data), 1.0/fps)
    
    # Ambil hanya frekuensi positif
    positive_freq_idx = fft_freq > 0
    fft_freq = fft_freq[positive_freq_idx]
    fft_magnitude = np.abs(fft_data[positive_freq_idx])
    
    # Smooth magnitude dengan moving average
    if len(fft_magnitude) > 5:
        fft_magnitude = np.convolve(fft_magnitude, np.ones(5)/5, mode='same')
    
    # Filter frekuensi dalam range jantung (0.67-4.0 Hz atau 40-240 BPM)
    valid_idx = (fft_freq >= 0.67) & (fft_freq <= 4.0)
    fft_freq = fft_freq[valid_idx]
    fft_magnitude = fft_magnitude[valid_idx]
    
    if len(fft_magnitude) == 0:
        return 0
    
    # Cari frekuensi dengan magnitude tertinggi
    max_idx = np.argmax(fft_magnitude)
    dominant_freq = fft_freq[max_idx]
    
    # Konversi ke BPM
    bpm = dominant_freq * 60
    return bpm

# Fungsi untuk estimasi BPM menggunakan peak detection
def estimate_bpm_peaks(signal_data, fps):
    """
    Estimasi BPM menggunakan deteksi puncak (peak detection) dengan kalman smoothing
    """
    if len(signal_data) < fps * 2:  # Minimal 2 detik data
        return 0
    
    # Normalisasi sinyal
    signal_norm = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-9)
    
    # Deteksi puncak dengan parameter yang lebih ketat
    # distance = jarak minimum antar puncak (dalam frames)
    # untuk 40-240 BPM, jarak minimum = fps/(240/60) = fps/4
    min_distance = max(int(fps / 4), 5)
    
    try:
        peaks, properties = find_peaks(signal_norm, distance=min_distance, prominence=0.3)
    except:
        return 0
    
    if len(peaks) < 2:
        return 0
    
    # Hitung jarak antar puncak
    peak_distances = np.diff(peaks)
    
    # Filter outliers menggunakan median absolute deviation
    median_distance = np.median(peak_distances)
    mad = np.median(np.abs(peak_distances - median_distance))
    
    # Abaikan jarak yang terlalu jauh dari median
    valid_distances = peak_distances[
        np.abs(peak_distances - median_distance) <= 2 * mad
    ]
    
    if len(valid_distances) < 1:
        return 0
    
    # Gunakan median untuk robustness
    avg_distance = np.median(valid_distances)
    
    # Konversi ke BPM
    bpm = (fps / avg_distance) * 60
    
    # Filter BPM yang tidak realistis
    if bpm < 40 or bpm > 240:
        return 0
    
    return bpm

# Fungsi untuk ekstraksi spatial averaging dari ROI
def extract_rgb_signal(roi):
    """
    Ekstraksi nilai RGB rata-rata dari ROI (spatial averaging)
    """
    if roi.size == 0:
        return 0, 0, 0
    
    # Hitung rata-rata untuk setiap channel
    b_avg = np.mean(roi[:, :, 0])
    g_avg = np.mean(roi[:, :, 1])
    r_avg = np.mean(roi[:, :, 2])
    
    return r_avg, g_avg, b_avg

# Main program
def main():
    # Inisialisasi variabel
    fps = 30
    time_window = 10  # Window 10 detik untuk analisis
    frame_buffer_limit = time_window * fps
    
    # Buffer untuk menyimpan sinyal RGB
    r_buffer = deque(maxlen=frame_buffer_limit)
    g_buffer = deque(maxlen=frame_buffer_limit)
    b_buffer = deque(maxlen=frame_buffer_limit)
    
    # Buffer untuk BPM history (untuk smoothing) - diperpanjang untuk stabilitas
    bpm_history = deque(maxlen=15)
    bpm_fft_history = deque(maxlen=10)
    bpm_peaks_history = deque(maxlen=10)
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Variabel untuk mengukur FPS aktual
    frame_count = 0
    start_time = time.time()
    actual_fps = fps
    
    print("=== Real-time rPPG Heart Rate Monitor ===")
    print("Tekan 'q' untuk keluar")
    print("Posisikan wajah Anda di depan kamera...")
    print()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Gagal membaca frame dari webcam")
            break
        
        frame_count += 1
        
        # Hitung FPS aktual setiap 30 frame
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            actual_fps = 30 / elapsed_time
            actual_fps = max(20, min(actual_fps, 60))  # Clamp ke range realistis
            start_time = time.time()
        
        # Convert BGR to RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi wajah
        results = face_detection.process(rgb_frame)
        
        # Proses jika wajah terdeteksi
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                
                # Konversi koordinat relatif ke absolut
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Scaling factor untuk fokus ke area dahi/pipi
                scaling_factor = 0.8
                margin_x = 10
                
                # Hitung ROI (fokus ke bagian tengah wajah)
                roi_x = max(0, x + margin_x)
                roi_y = max(0, y)
                roi_w = min(width - 2 * margin_x, w - roi_x)
                roi_h = int(height * scaling_factor)
                
                # Extract ROI
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                
                # Ekstraksi sinyal RGB (spatial averaging)
                if roi.size > 0:
                    r_avg, g_avg, b_avg = extract_rgb_signal(roi)
                    r_buffer.append(r_avg)
                    g_buffer.append(g_avg)
                    b_buffer.append(b_avg)
                
                # Gambar ROI pada frame
                cv2.rectangle(frame, (roi_x, roi_y), 
                            (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                cv2.putText(frame, "ROI", (roi_x, roi_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Proses sinyal jika buffer sudah cukup (minimum 5 detik untuk stabilitas)
                if len(g_buffer) >= fps * 5:  # Minimal 5 detik data
                    # Konversi buffer ke numpy array
                    g_signal = np.array(g_buffer)
                    
                    # 1. Detrending
                    g_detrended = detrend_signal(g_signal, window_size=int(actual_fps * 1.5))
                    
                    # 2. Bandpass filter
                    g_filtered = bandpass_filter(g_detrended, actual_fps)
                    
                    # 3. Estimasi BPM menggunakan FFT
                    bpm_fft = estimate_bpm_fft(g_filtered, actual_fps)
                    
                    # 4. Estimasi BPM menggunakan peak detection (sebagai validasi)
                    bpm_peaks = estimate_bpm_peaks(g_filtered, actual_fps)
                    
                    # Tambahkan ke history jika valid
                    if 40 <= bpm_fft <= 240:
                        bpm_fft_history.append(bpm_fft)
                    if 40 <= bpm_peaks <= 240:
                        bpm_peaks_history.append(bpm_peaks)
                    
                    # Gunakan median dari history untuk robustness
                    if len(bpm_fft_history) > 0 and len(bpm_peaks_history) > 0:
                        bpm_fft_median = np.median(list(bpm_fft_history))
                        bpm_peaks_median = np.median(list(bpm_peaks_history))
                        bpm = (bpm_fft_median + bpm_peaks_median) / 2
                    elif len(bpm_fft_history) > 0:
                        bpm = np.median(list(bpm_fft_history))
                    elif len(bpm_peaks_history) > 0:
                        bpm = np.median(list(bpm_peaks_history))
                    else:
                        bpm = 0
                    
                    # Smoothing BPM dengan exponential moving average
                    if bpm > 0:
                        bpm_history.append(bpm)
                        # Gunakan weighted average: lebih baru = lebih penting
                        weights = np.exp(np.linspace(-2, 0, len(bpm_history)))
                        weights /= weights.sum()
                        bpm_smoothed = np.average(list(bpm_history), weights=weights)
                    else:
                        bpm_smoothed = np.median(bpm_history) if len(bpm_history) > 0 else 0
                    
                    # Tampilkan BPM pada frame
                    if bpm_smoothed > 0:
                        cv2.putText(frame, f"Heart Rate: {bpm_smoothed:.0f} BPM", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  (0, 255, 0), 2)
                        cv2.putText(frame, f"Status: Measuring", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Heart Rate: -- BPM", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  (0, 0, 255), 2)
                        cv2.putText(frame, "Status: Collecting data...", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                  (0, 255, 255), 2)
                else:
                    # Tampilkan status collecting
                    progress = (len(g_buffer) / (fps * 5)) * 100
                    cv2.putText(frame, f"Initializing: {progress:.0f}%", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                              (0, 255, 255), 2)
        else:
            # Tidak ada wajah terdeteksi
            cv2.putText(frame, "No face detected", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Tampilkan FPS
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, h - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tampilkan buffer status
        buffer_status = f"Buffer: {len(g_buffer)}/{frame_buffer_limit}"
        cv2.putText(frame, buffer_status, (10, h - 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tampilkan frame
        cv2.imshow('Real-time rPPG Heart Rate Monitor', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    print("\nProgram selesai.")

if __name__ == "__main__":
    main()