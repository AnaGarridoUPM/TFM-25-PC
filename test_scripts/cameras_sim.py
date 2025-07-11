# base64 - otro payload (mas grande)
# I dont know why I created this script

import os
import base64
import json
import time
import threading
from logger_config import logger
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
import numpy as np
import open3d as o3d 
import random 

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
'''
K = [
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0]
]
'''

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

# Function to encode and transmit files via MQTT msg
def encode_png_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Calculate size of JSON-encoded payload in bytes
def get_message_size(payload):
    return len(json.dumps(payload))

# Function to construct and send message to broker hosted in HiveMQ
def build_publish_encoded_msg(client, camera_name, k, color_name, encoded_color_file, depth_name, encoded_depth_file, dataset_id, container_name, total_cameras):
    dt_now = datetime.now(tz=timezone.utc) 
    send_ts = round(dt_now.timestamp() * 1000) # unicidad del mensaje 

    payload = {
        "frame_color_name": color_name,
        "enc_c": encoded_color_file,
        "frame_depth_name": depth_name,
        "enc_d": encoded_depth_file,
        "K": k,
        "send_ts": send_ts, # UTC timestamp
        "container_name": container_name,
        "total_cameras": total_cameras
    }

    # Calculate message size
    message_size = get_message_size(payload)
    logger.info(f"Message size base64: {message_size} bytes. ENVIADO.")

    client.publish(MQTT_TOPIC_CAM, json.dumps(payload), qos=MQTT_QOS) # qos=0 best effort
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
        encoded_color_file = encode_png_to_base64(os.path.join(path_color, chosen_color_frame))
        encoded_depth_file = encode_png_to_base64(os.path.join(path_depth, chosen_depth_frame))
        build_publish_encoded_msg(client, camera_name, k_list, chosen_color_frame, encoded_color_file, chosen_depth_frame, encoded_depth_file, dataset_id, container_name, total_cameras)
    
    logger.info(f"[Sending all frames of camera {camera_name} END")
    
# Function to control the flow and send frames and files 
# nota: base_directory = data/cameras
def start_cam_simulation(client, base_directory, dataset_id, container_name, total_cameras, downsample_factor, send_freq = 3):
    time_start = time.perf_counter()
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
            # Code block to measure
            sum(range(1000000))
            end_framebatch = time.perf_counter()
            logger.info(f"Execution time: {end_framebatch - time_start:.6f} seconds")
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

def get_sequence_name():
    logger.info("Please enter name of the sequence \n(no spaces, no simbols, only letters and numbers):")
    container_name = input()
    logger.info("Please enter number of cameras used:")
    total_cameras = input()
    logger.info("Please enter downsampling factor or skip (press ENTER):")
    downsample_factor = input()
    return container_name, total_cameras, downsample_factor

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
        time.sleep(4) # wait for connection setup to complete 

        client.loop_start()
    except Exception as e:
        logger.error(f"Could not connect to broker: {e}")
    else:
        # Starting data publication
        logger.info("Connected to HiveMQ Cloud MQTT.")
        base_directory = './data/frames/'
        dataset_id = 1 # no quitar (recycling mikel scripts - point clouds)
        container_name, total_cameras, downsample_factor = get_sequence_name()
        x = input("Press ENTER to start") 
        start_cam_simulation(client, base_directory, dataset_id, container_name, total_cameras, downsample_factor, send_freq=SEND_FREQUENCY) # send frames
        logger.info("Simulation ended")