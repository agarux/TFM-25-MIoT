# Source --> https://github.com/mikeligUPM/tfm_edgecloud_registrator
# CAMERAS SIMULATION. This script sends all frames of x camera to the broker under MQTT - TOPIC '1cameraframes'
# OJO: las imagenes tienen que tener en su nombre 'color' o 'depth' y numero de frame
# *--data
# *     |--cameras
# *             |--000442922112    
# *                         |--color
# *                         |--depth
# External script "./rename_files.py". ARGS: full path to the directory that contains all files EXAMPLE PATH: "./data/cameras/000442922112/depth_frames/"
#* Maybe it is not necessary to previously change the name of the frames and do it in the payload 
#* what I need at the server: camID_color_frame0003.png or camID_depth_frame0021.png
#* what may i have: camID_1920x1080_yuv420p8le_frame_0219.png or camID_1920x1080_gray_16bit_frame_0045.png
# necesario diezmar by default 3 FPS

import os
import time
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
'''
K = [
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0]
]
'''

### TEST PNG COMPRESSION - SENDER ###
def calculate_metrics_rgb (img_original, img_compresed):
    mse = np.mean((img_original - img_compresed)**2)
    logger.info(f"[RGB] Mean Square Error: {mse:.2f}")
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    logger.info(f"[RGB] PSNR: {psnr} dB")

def calculate_metrics_depth (img_original, img_compresed):
    mse = np.mean((img_original - img_compresed)**2)
    logger.info(f"[Depth] Mean Square Error: {mse:.2f}")
    if mse == 0:
        return float('inf')
    max_pixel = 65536.0
    psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    logger.info(f"[Depth] PSNR: {psnr} dB")



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

# read file and compressed it: OpenCV 
def read_png(file_path_c, file_path_d):
    # RGB file
    img_cv_c = cv2.imread(file_path_c, cv2.IMREAD_UNCHANGED)
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, encoded_c_img = cv2.imencode('.png', img_cv_c, encode_param)
    # Depth file 
    img_cv_d = cv2.imread(file_path_d, cv2.IMREAD_UNCHANGED)
    assert img_cv_d.dtype == np.uint16, "Expected 16-bit depth image"
    success, encoded_d_img = cv2.imencode('.png', img_cv_d, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not success:
        raise ValueError("Failed to encode PNG")
    
    frame_color_decoded = cv2.imdecode(np.frombuffer(encoded_c_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    calculate_metrics_rgb(img_cv_c, frame_color_decoded)
    calculate_metrics_depth(img_cv_d, encoded_d_img)

    return encoded_c_img.tobytes(), encoded_d_img.tobytes()

# Function to construct and send message to broker hosted in HiveMQ
def build_publish_encoded_msg(client, camera_name, k, color_name, cv_c_img, depth_name, depth_f, container_name, total_cameras):
    dt_now = datetime.now(tz=timezone.utc) 
    send_ts = round(dt_now.timestamp() * 1000)
    
    payload = {
        "frame_color_name": color_name,
        "color_file": cv_c_img,
        "frame_depth_name": depth_name,
        "depth_file": depth_f,
        "K": k,
        "send_ts": send_ts, # UTC timestamp
        "container_name": container_name,
        "total_cameras": total_cameras
    }
    
    cbor_payload = cbor2.dumps(payload)
    logger.info(f"Message size CBOR: {len(cbor_payload)} bytes. ENVIADO.")

    client.publish(MQTT_TOPIC_CAM, cbor_payload, qos=MQTT_QOS)
    logger.info(f"[TS] SEQUENCE: {container_name}. Camera [{camera_name}] sent message to BROKER, color: {color_name}, depth {depth_name}, time {send_ts}")
    
# PARAMETER: elegir DIEZMADO!!! 
def process_frames_of_a_camera(client, k_dict, camera_name_path, container_name, total_cameras, downsample_factor): 
    camera_name = os.path.basename(camera_name_path) # 0004422112
    logger.info(f"Sending all frames of camera {camera_name} INIT")

    directories = [os.path.join(camera_name_path, d) for d in os.listdir(camera_name_path) if os.path.isdir(os.path.join(camera_name_path, d))]

    for dir in directories:
        '''
        if 'grey' in os.path.basename(dir):
            path_depth = dir
            depth_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_1920x1080_gray_16bit_"))
            depth_frames = depth_frames[::factor]  # diezmado: 1 de cada factor 
        '''
        if 'color' in os.path.basename(dir):
            path_color = dir
            color_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_color_"))
            color_frames = color_frames[::downsample_factor]  # diezmado: 1 de cada factor (default = 6)
        if 'depth' in os.path.basename(dir):
            path_depth = dir
            depth_frames = sorted(f for f in os.listdir(dir) if f.startswith(f"{camera_name}_depth_"))
            depth_frames = depth_frames[::downsample_factor]  # diezmado: 1 de cada factor 

    if isinstance(k_dict, dict): # if k_dict es un dicc o lista 
        k_list = k_dict[camera_name]
    elif isinstance(k_dict, list):
        k_list = k_dict

    # OpenCV compression 
    for chosen_color_frame, chosen_depth_frame in zip(sorted(color_frames), sorted(depth_frames)):
        opencv_c, opencv_d = read_png(os.path.join(path_color, chosen_color_frame), os.path.join(path_depth, chosen_depth_frame))
        build_publish_encoded_msg(client, camera_name, k_list, chosen_color_frame, opencv_c, chosen_depth_frame, opencv_d, container_name, total_cameras)
    
    logger.info(f"[Sending all frames of camera {camera_name} END")
    
# Function to control the flow and send frames and files 
# nota: base_directory = data/cameras
def start_cam_simulation(client, base_directory, container_name, total_cameras, downsample_factor, send_freq = 3):
    x = input("Press ENTER to start") 
    time_start = time.perf_counter()
    exit_sim = False # ESC
    filepath = 'cam_params.json' # extrinsic and intrinsic parameters 
    k_dict = create_k_dict_by_camera(filepath)
    try:
        while not exit_sim:
            camera_name_directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
            threads = []  
            for cam_name_dir in camera_name_directories: 
                thread = threading.Thread(target=process_frames_of_a_camera, args=(client, k_dict, cam_name_dir, container_name, total_cameras, downsample_factor))
                threads.append(thread)
                thread.start()
                time.sleep(0.1) 
                    
            for thread in threads: # wait threads
                thread.join()
                break
            
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

def get_user_param():
    while True:
        # Prompt for container name and validate
        logger.info("Please enter name of the sequence (no spaces, no symbols, only letters and numbers):")
        container_name = input().strip()
        if not container_name.isalnum():
            logger.warning("Invalid name. Please only use letters and numbers.")
            continue
        
        # Prompt for the number of cameras and validate
        logger.info("Please enter number of cameras used:")
        try:
            total_cameras = int(input())
            if total_cameras <= 0:
                logger.warning("Number of cameras must be a positive integer.")
                continue
        except ValueError:
            logger.warning("Invalid number of cameras. Please enter a valid integer.")
            continue
        
        # Prompt for downsampling factor or skip (default 6)
        logger.info("Please enter downsampling factor or skip (by default 3), press ENTER:")
        downsample_factor_input = input().strip()
        
        if downsample_factor_input == "":
            downsample_factor = 3  # default value
        else:
            try:
                downsample_factor = int(downsample_factor_input)
                if downsample_factor <= 0:
                    logger.warning("Downsampling factor must be a positive integer.")
                    continue
            except ValueError:
                logger.warning("Invalid downsampling factor. Please enter a valid integer.")
                continue
        
        return container_name, total_cameras, downsample_factor


# MAIN 
if __name__ == "__main__":
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) # Connection to MQTT broker
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
        
        #* Real time implementation - this code must be modificated a lot depending on how the cameras work --> threads?¿?¿?
        base_directory = './data/frames/' 
        container_name, total_cameras, downsample_factor = get_user_param() # variables with their correct types (string, int, int)
        
        start_cam_simulation(client, base_directory, container_name, total_cameras, downsample_factor, send_freq=SEND_FREQUENCY) # send frames
        logger.info("Simulation ended")


        