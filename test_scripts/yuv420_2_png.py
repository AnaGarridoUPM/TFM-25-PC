import numpy as np
import cv2
import os

# YUV file parameters
width = 1920
height = 1080
frame_size = width * height * 3 // 2  # YUV 4:2:0

def read_yuv_file(filename):
    with open(filename, 'rb') as f:
        return f.read()

def yuv_to_rgb(yuv_data, width, height):
    Y = np.frombuffer(yuv_data[:width * height], dtype=np.uint8).reshape((height, width))
    U = np.frombuffer(yuv_data[width * height:width * height + (width // 2) * (height // 2)],
                      dtype=np.uint8).reshape((height // 2, width // 2))
    V = np.frombuffer(yuv_data[width * height + (width // 2) * (height // 2):],
                      dtype=np.uint8).reshape((height // 2, width // 2))

    U = cv2.resize(U, (width, height), interpolation=cv2.INTER_LINEAR)
    V = cv2.resize(V, (width, height), interpolation=cv2.INTER_LINEAR)

    yuv_image = np.stack((Y, U, V), axis=-1)
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return rgb_image

def save_yuv_frames_to_png(filename, width, height, output_dir, base_name):
    raw_data = read_yuv_file(filename)
    total_size = len(raw_data)
    frame_count = total_size // frame_size
    print(f"Total frames found: {frame_count}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame_idx in range(frame_count):
        frame_start = frame_idx * frame_size
        frame_end = frame_start + frame_size
        frame_data = raw_data[frame_start:frame_end]

        rgb_frame = yuv_to_rgb(frame_data, width, height)

        output_filename = f"{base_name}_color_f{frame_idx + 1:04d}.png"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, rgb_frame)

        del rgb_frame

    print("Todas las imágenes han sido guardadas en la carpeta:", output_dir)

# --- CONFIGURACION MANUAL ---
file_path = 'rgb/001070614312_1920x1080_yuv420p8le.yuv'
output_directory = 'color/001070614312'
base_filename = '001070614312'

save_yuv_frames_to_png(file_path, width, height, output_directory, base_filename)