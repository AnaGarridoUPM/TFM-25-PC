# Script to measure size and time of execution
# Put this script in the same folder as the data folder with the frames 
# Change the server side to allow receiving messages as raw, cbor or base64 encoded

import base64
import os
import time
import random 
import threading
from logger_config import logger
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
import json
import open3d as o3d 
import numpy as np
import cbor2
import cv2

# MyhiveMQTT - serverless - 10GB/month - FREE: only 1 cluster at the same time 
MQTT_BROKER = '7c9990070e35402ea3c6ad7ccf724e0b.s1.eu.hivemq.cloud'
MQTT_PORT = 8883
MQTT_QOS = 1 
MQTT_TOPIC_CAM = '1cameraframes'
SEND_FREQUENCY = 1  # Time in seconds between sending messages
# user (cameras)
USERNAME = 'user_cameras_tfm25'
PASSWORD = 'camerasK2425'

logger.info(f"Camera simulation started with:\nBROKER_IP: {MQTT_BROKER}\nBROKER_PORT: {MQTT_PORT}\nSEND_FREQUENCY: {SEND_FREQUENCY}")

# TEST 
def get_size(obj):
    if isinstance(obj, str):
        return len(obj.encode('utf-8'))
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, (int, float)):
        return 8  # conservative estimate (float64 or int64)
    elif isinstance(obj, list):
        return sum(get_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_size(k) + get_size(v) for k, v in obj.items())
    else:
        return 0  # fallback for unsupported types


# camera parameters
def create_k_dict_by_camera(filepath) -> dict:
    k_dict = {}
    K = np.eye(3)
    with open(filepath, "r") as f:
        data = json.load(f)
        for _, camera in enumerate(data["cameras"]):
            # Extract camera parameters
            resolution = camera["Resolution"]
            focal = camera["Focal"]
            principal_point = camera["Principle_point"]
            camera_name = camera["Name"]
            # Create PinholeCameraIntrinsic object
            K = o3d.camera.PinholeCameraIntrinsic(
                width=resolution[0],
                height=resolution[1],
                fx=focal[0],
                fy=focal[1],
                cx=principal_point[0],
                cy=principal_point[1]
            )
            k_dict[camera_name] = K.intrinsic_matrix.tolist()
    return k_dict

'''
# TEST leer datos 
def test_png(file_path_c, file_path_d):
    level = 7
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), level]

    # COMPRESION DE IMAGEN CON OPENCV
    # Imagen RGB
    img_cv_c = cv2.imread(file_path_c, cv2.IMREAD_UNCHANGED)
    success_c, cv_c_img = cv2.imencode('.png', img_cv_c, encode_param)
    if not success_c:
        raise ValueError("Failed to encode PNG for color image")
    
    # Imagen de profundidad (depth)
    img_cv_d = cv2.imread(file_path_d, cv2.IMREAD_UNCHANGED)
    assert img_cv_d.dtype == np.uint16, "Expected 16-bit depth image"
    success_d, cv_d_img = cv2.imencode('.png', img_cv_d, encode_param)
    if not success_d:
        raise ValueError("Failed to encode PNG for depth image")
    
    # CODIFICACION EN BASE64
    base64_c = base64.b64encode(cv_c_img).decode('utf-8')
    base64_d = base64.b64encode(cv_d_img).decode('utf-8')

    return cv_c_img.tobytes(), cv_d_img.tobytes(), base64_c, base64_d
'''

def calculate_psnr(img1, img2):
    mse=np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_psnr_d(img1, img2):
    mse=np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 65536.0
    
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def test_png(file_path_c, file_path_d):
    start = time.perf_counter()
    quality=10
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ext = '.png'

    # Read original images

    img_cv_c = cv2.imread(file_path_c, cv2.IMREAD_UNCHANGED)
    img_cv_d = cv2.imread(file_path_d, cv2.IMREAD_UNCHANGED)
    assert img_cv_d.dtype == np.uint16, "Expected 16-bit depth image"

    # Encode images as JPEG
    success_c, comp_c = cv2.imencode(ext, img_cv_c, encode_param)
    if not success_c:
        raise ValueError("Failed to encode color image")

    success_d, comp_d = cv2.imencode('.png', img_cv_d, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not success_d:
        raise ValueError("Failed to encode depth image")

    end = time.perf_counter()
    # Decode compressed images for PSNR calculation
    dec_c = cv2.imdecode(comp_c, cv2.IMREAD_UNCHANGED)
    dec_d = cv2.imdecode(comp_d, cv2.IMREAD_UNCHANGED)

    # Calculate PSNR
    psnr_c = calculate_psnr(img_cv_c, dec_c)
    psnr_d = calculate_psnr_d(img_cv_d, dec_d)
    tiempo = end-start
    logger.info(f"PSNR RGB {psnr_c}, quality {quality}. Time compression {tiempo}")
    logger.info(f"PSNR depth {psnr_d}, quality {quality}.")

    #cv2.imwrite("compressed_rgb_image.jpg", dec_c)

    # Base64 encode compressed images
    base64_c = base64.b64encode(comp_c).decode('utf-8')
    base64_d = base64.b64encode(comp_d).decode('utf-8')

    return comp_c.tobytes(), comp_d.tobytes(), base64_c, base64_d


# Function to construct and send message to broker hosted in HiveMQ
def build_publish_encoded_msg(client, camera_name, k, color_name, depth_name, dataset_id, container_name, total_cameras, cv_c_img, cv_d_img, base64_c, base64_d):
    dt_now = datetime.now(tz=timezone.utc) 
    send_ts = round(dt_now.timestamp() * 1000) # unicidad del mensaje 
    _,_,num_frame = (color_name.split('.')[0]).split('_') # name frame: cameranum_typeframe_numberframe.png

    payload_raw = {
        "enc_c": cv_c_img,
        "enc_d": cv_d_img,
        "camera_id": camera_name,
        "num_frame": num_frame,
        "K": k,
        "send_ts": send_ts, # UTC timestamp
        "container_name": container_name,
        "total_cameras": total_cameras
    }
    
    payload_cbor = {
        "enc_c": cv_c_img,
        "enc_d": cv_d_img,
        "camera_id": camera_name,
        "num_frame": num_frame,
        "K": k,
        "send_ts": send_ts, # UTC timestamp
        "container_name": container_name,
        "total_cameras": total_cameras
    }

    payload_base64 = {
        "enc_c": base64_c,
        "enc_d": base64_d,
        "camera_id": camera_name,
        "num_frame": num_frame,
        "K": k,
        "send_ts": send_ts, # UTC timestamp
        "container_name": container_name,
        "total_cameras": total_cameras
    }

    '''
    # Measure and print sizes - RAW separately but now together just to see 
    total_raw_size = 0
    logger.info("\n=== RAW data ===")
    for key, value in payload_raw.items():
        field_size = get_size(value)
        total_raw_size += field_size
        logger.info(f"{key:20}: {field_size} bytes")
    logger.info("=================")
    logger.info(f"Estimated total raw size: {total_raw_size} bytes\n")
    '''
    # Measure and print sizes
    total_cv_size = 0
    logger.info("\n=== CV compression ===")
    for key, value in payload_cbor.items():
        field_size = get_size(value)
        total_cv_size += field_size
        logger.info(f"{key:20}: {field_size} bytes")
    logger.info("============================")
    logger.info(f"Message size CV: {total_cv_size} bytes")
    cbor_payload = cbor2.dumps(payload_cbor)
    logger.info(f"Message size CBOR: {len(cbor_payload)} bytes. ENVIADO.\n")
    #cbor_comppayload = json.dumps(payload_cbor)
    #logger.info(f"Message size CBOR + raw data: {len(cbor_comppayload)} bytes. ENVIADO.\n")
   
    # Measure and print sizes
    total_b64_size = 0
    logger.info("\n=== base64 encoding ===")
    for key, value in payload_base64.items():
        field_size = get_size(value)
        total_b64_size += field_size
        logger.info(f"{key:20}: {field_size} bytes")
    logger.info("============================")
    logger.info(f"Message size CV: {total_b64_size} bytes")
    json_payload = json.dumps(payload_base64)
    logger.info(f"Message size JSON: {len(json_payload)} bytes. ENVIADO.\n")
    cb_payload = cbor2.dumps(payload_base64)
    logger.info(f"Message size CBOR: {len(cb_payload)} bytes. ENVIADO.\n")

    #client.publish(MQTT_TOPIC_CAM, raw_data_c, qos=MQTT_QOS)
    #client.publish(MQTT_TOPIC_CAM, raw_data_d, qos=MQTT_QOS)
    #print("todo junto")
    #client.publish(MQTT_TOPIC_CAM, payload_raw, qos=MQTT_QOS)
    #print("cbor")
    #client.publish(MQTT_TOPIC_CAM, cbor_payload, qos=MQTT_QOS)
    #print("json")
    client.publish(MQTT_TOPIC_CAM, json_payload, qos=MQTT_QOS)
    logger.info(f"[TS] SEQUENCE: {container_name}. Camera [{camera_name}] sent message to BROKER, color: {color_name}, depth {depth_name}, time {send_ts}")


# PARAMETER: elegir DIEZMADO!!! 
def process_frames_of_a_camera(client, k_dict, camera_name_path, dataset_id, container_name, total_cameras, downsample_factor): 
    if downsample_factor == "":
        factor = 3
    else:
        factor = int(downsample_factor)

    camera_name = os.path.basename(camera_name_path) # 0004422112
    logger.info(f"Sending all frames of camera {camera_name} INIT")

    directories = [os.path.join(camera_name_path, d) for d in os.listdir(camera_name_path) if os.path.isdir(os.path.join(camera_name_path, d))]

    for dir in directories:
        if 'color' in os.path.basename(dir):
            path_color = dir
            color_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_color_"))
            color_frames = color_frames[::factor]  # diezmado: 1 de cada factor (default = 3)
        if 'depth' in os.path.basename(dir):
            path_depth = dir
            depth_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_depth_"))
            depth_frames = depth_frames[::factor]  # diezmado: 1 de cada factor 

    if isinstance(k_dict, dict): # if k_dict es un dicc o lista 
        k_list = k_dict[camera_name]
    elif isinstance(k_dict, list):
        k_list = k_dict

    for chosen_color_frame, chosen_depth_frame in zip(sorted(color_frames), sorted(depth_frames)):
        encoded_c_img, encoded_d_img, base64_c, base64_d  = test_png(os.path.join(path_color, chosen_color_frame), os.path.join(path_depth, chosen_depth_frame))
        build_publish_encoded_msg(client, camera_name, k_list, chosen_color_frame, chosen_depth_frame, dataset_id, container_name, total_cameras, encoded_c_img, encoded_d_img, base64_c, base64_d)
    
    logger.info(f"[Sending all frames of camera {camera_name} END")

    
# Function to control the flow and send frames and files 
# nota: base_directory = data/cameras
def start_cam_simulation(client, base_directory, dataset_id, container_name, total_cameras, downsample_factor, send_freq = 3):
    x = input("Press ENTER to start") 
    exit_sim = False # ESC
    filepath = 'cam_params.json'
    k_dict = create_k_dict_by_camera(filepath)

    try:
        while not exit_sim:
            camera_name_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
            threads = []  
            for cam_name_dir in camera_name_directories: 
                thread = threading.Thread(target=process_frames_of_a_camera, args=(client, k_dict, cam_name_dir, dataset_id, container_name, total_cameras, downsample_factor))
                threads.append(thread)
                thread.start()
                time.sleep(0.1) 
                    
            for thread in threads: # wait threads
                thread.join()
            
            time.sleep(send_freq)  # Esperar N segundos antes de comenzar nuevamente
            x = input("continue? \n")    
    except KeyboardInterrupt:
        exit_sim = True

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(MQTT_TOPIC_CAM, MQTT_QOS)
    else:
        logger.error(f"Error connecting to broker, code: {rc}")  

# MQTT Publish function 
def on_publish(client, userdata, mid, reason_codes=None, properties=None):
    logger.info(f"Message published successfully with MID: {mid}")

# MAIN 
if __name__ == "__main__":
    try:
        # Connection to MQTT broker
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set()

        client.on_connect = on_connect
        client.on_publish = on_publish

        client.connect(MQTT_BROKER, MQTT_PORT)

        client.loop_start()
    except Exception as e:
        logger.error(f"Could not connect to broker: {e}")
    else:
        # Starting data publication
        logger.info("Connected to HiveMQ Cloud MQTT.")
        base_directory = './data/frames/'
        dataset_id = 1 # no quitar (recycling mikel scripts - point clouds)
        container_name = "testsizes"
        total_cameras = 3
        downsample_factor = 3
        start_cam_simulation(client, base_directory, dataset_id, container_name, total_cameras, downsample_factor, send_freq=SEND_FREQUENCY) # send frames
        logger.info("Simulation ended")