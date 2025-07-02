import numpy as np
import cv2
import os

# Parámetros del archivo
width = 1920
height = 1080
frame_size = width * height * 2  # 16 bits = 2 bytes por píxel

def read_gray16_yuv(filename):
    with open(filename, 'rb') as f:
        return f.read()

def is_split_frame(img, threshold=500):
    """Detecta si la imagen está partida por la mitad horizontalmente."""
    h = img.shape[0]
    top = img[:h//2, :]
    bottom = img[h//2:, :]
    
    # Comparar las filas adyacentes alrededor del corte
    edge_diff = np.abs(top[-1, :].astype(np.int32) - bottom[0, :].astype(np.int32))
    score = np.mean(edge_diff)

    return score > threshold

def save_frames_from_gray16_yuv(filename, width, height, output_dir, base_name):
    raw_data = read_gray16_yuv(filename)
    total_frames = len(raw_data) // frame_size
    print(f"Total frames encontrados: {total_frames}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(total_frames):
        start = idx * frame_size
        end = start + frame_size
        frame_data = raw_data[start:end]

        frame = np.frombuffer(frame_data, dtype=np.uint16).reshape((height, width))

        # Detectar y corregir si está "partida" horizontalmente
        if is_split_frame(frame):
            frame = np.vstack((frame[height//2:], frame[:height//2]))

        output_filename = f"{base_name}_depth_f{idx + 1:04d}.png"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, frame)

        del frame

    print("Imágenes guardadas en:", output_dir)

# --- CONFIGURACION MANUAL ---
camara = '001070614312'

file_path = f'gray/{camara}_1920x1080_gray_16bit.yuv'
output_directory = f'data/frames/{camara}/depth_frames'


save_frames_from_gray16_yuv(file_path, width, height, output_directory, camara)