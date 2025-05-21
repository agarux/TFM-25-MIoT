# Source --> https://github.com/mikeligUPM/tfm_edgecloud_registrator/tree/main
#        --> mmsegmentation # Copyright (c) OpenMMLab. All rights reserved.

import base64
import json
import threading
import numpy as np
import cv2
import open3d as o3d 
from threading import Timer
import paho.mqtt.client as mqtt
from mmseg.apis import inference_model, init_model, show_result_pyplot
from logger_config import logger # logs 
from helper_funs import get_config
from registrator_icp_ransac import icp_p2p_registration_ransac
from blob_handler import save_and_upload_pcd

# MyhiveMQTT - serverless - 10GB/month - FREE: only 1 cluster at the same time 
MQTT_BROKER = '7c9990070e35402ea3c6ad7ccf724e0b.s1.eu.hivemq.cloud'
MQTT_PORT = 8883
MQTT_QOS = 1 
MQTT_TOPIC_CAM = '1cameraframes'
SEND_FREQUENCY = 1  # Time in seconds between sending messages
# server
USERNAME_S = 'user_server_tfm25'
PASSWORD_S = 'serverM2425'
# OpenMMLab - MMSegmentation 
CONF_FILE = './segmentation/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py' # model PSPNET, dataset: ADE20k
CHKP_FILE = './segmentation/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth' # trained model 

#* Set mmsegmentation model ONLY 1 TIME
model = init_model(CONF_FILE, CHKP_FILE, device = 'cpu') # MODEL in CPU 
#* AVOID MQTT MSG DUPLICATION
processed_timestamps = set()
received_frames_dict = {}
received_frames_lock = threading.Lock()
batch_timeout = 120 # seconds, 2 min?

#* AUX function - decode files from binary to numpy array, by default BGR 
def decode_files(color_enc, depth_enc):
    ## RGB file decode  
    decode_c_file_binary = base64.b64decode(color_enc)
    if len(decode_c_file_binary) % 2 != 0:
        decode_c_file_binary += b'\x00'
        logger.debug(f"Adjusted len color_image_data: {len(decode_c_file_binary)}\n")
    frame_color_decoded = cv2.imdecode(np.frombuffer(decode_c_file_binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame_color_decoded is None:
        logger.error("Decoded color image is None. Error loading image or corverting color image to RGB. Check the data format")
    ## Depth file decode  
    decode_d_file_binary = base64.b64decode(depth_enc)
    if len(decode_d_file_binary) % 2 != 0:
        decode_d_file_binary += b'\x00'
        logger.debug(f"Adjusted len depth_image_data: {len(decode_d_file_binary)}\n")
    frame_depth_decoded = cv2.imdecode(np.frombuffer(decode_d_file_binary, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
    if frame_depth_decoded is None:
        logger.error("Decoded depth image is None, check the data format")
    return frame_color_decoded, frame_depth_decoded

#* SECOND STEP: SIMPLE PC 1PC 4 1FRAME
def create_pc_from_enc_data(color_image, depth_image, K, target_ds):
    try:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    except Exception as e:
        logger.error(f"Error converting color image to RGB: {e}")
    depth_raw = o3d.geometry.Image((depth_image.astype(np.float32) / 1000.0))  # Dividir entre 1000 si está en mm
    color_raw = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    logger.debug(f"PCD len before downsampling {len(pcd.points)}")
    vox_size, _ = get_config(target_ds)
    logger.debug(f"[TEST] Voxel size: {vox_size}")
    pcd = pcd.voxel_down_sample(voxel_size=vox_size)
    logger.debug(f"PCD len after downsampling {len(pcd.points)}")
    pcd.estimate_normals()
    return pcd

# process frames of the same number
def process_frames(msg_frames_list, num_frame, container_name, K):
    logger.info(f"Processing batch of messages corresponding to number of frame: {num_frame}")
    
    #* PREPARE DATA ------------------------------------------------------------------------------------
    # auxiliary lists for 1 FRAME
    color_array = []
    depth_array = []  # decode frames (MQTTmsg: string --> np.array)
    seg_color_array = []
    seg_depth_array = []  # segmented info, to change object refer to documentation 
    pcd_list = []  # simple pc, no ICP algorithm implemented 
    target_ds = 1 

    for message in msg_frames_list: # processing el batch
        enc_c = message.get('enc_c') # tipo de varible: STR 
        enc_d = message.get('enc_d')
        frame_color_decoded, frame_depth_decoded = decode_files(enc_c, enc_d) # tipo de varible: NUMPY.NDARRAY 
        # Append images
        color_array.append(frame_color_decoded)
        depth_array.append(frame_depth_decoded)
    
    #* FIRST: segmentation & apply results to the sequence ----------------------------------------------
    results = inference_model(model, color_array)    
    for imagen, resultado, profun in zip(color_array, results, depth_array):
        # nota mental: si te guardas la imagen desde resultado el orden RGB no cambia, si lo haces desde el out_file si, PQ? no se, ole
        segmentation = show_result_pyplot(model, imagen, resultado, opacity = 1, draw_gt = False, show = False, with_labels=False, save_dir=None, out_file=None)  
        mask = np.all(segmentation == [150, 5, 61], axis=-1)  # boolean mask
        color_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA) # Set the alpha channel: 255 for people, 0 for the background
        color_img[:, :, 3] = mask.astype(np.uint8)*255
        profun[~mask] = 0 # Depth frame with black background 
        # save segmented images
        seg_color_array.append(color_img)
        seg_depth_array.append(profun)
    '''
    # chekipoins - BORRAR
    print(f"Size of the array of segmented color images: {len(seg_color_array)}") 
    print(f"Size of the array of segmented depth images: {len(seg_depth_array)}")
    '''
    i = 0
    #* SECOND: create simple 1pc for 1frame ----------------------------------------------------------------
    # TESTING, in real: delete save_and_upload_pcd
    for color_seg, depth_seg in zip(seg_color_array, seg_depth_array):
        i = i+1
        pc = create_pc_from_enc_data(color_seg, depth_seg, K, target_ds)
        '''
        # chekipoins - BORRAR
        save_and_upload_pcd(pc, f"simple_pc_camera_{i}_{num_frame}.ply", container_name)
        '''
        if pc is None:
            logger.error(f"Error creating point cloud for frame {num_frame} of camera {i}")
            continue
        logger.info(f"[TS] Frame [{num_frame}] PCD created for camera {i}")
        pcd_list.append(pc)
    '''
    # chekipoins - BORRAR
    print(f"Size of the array of simple point clouds: {len(pcd_list)}") 
    '''
    #* THIRD: ICP algorithm - fusion
    final_fused_point_cloud = icp_p2p_registration_ransac(pcd_list, target_ds)
    if final_fused_point_cloud is None:
        logger.info(f"Frame [{num_frame}] Final PCD is None. Please check error logs.")
    else:
        logger.debug(f"FRAME [{num_frame}] REGISTRATION SUCCESSFUL")
        reg_name = "icp_ransac" #reg_name = registration_names_from_id.get(target_registration, "unknown") RANSAC effectively filters out the noise in the data
        blob_name_reg = f"{container_name}_{num_frame}_{reg_name}.ply"
        save_and_upload_pcd(final_fused_point_cloud, blob_name_reg, container_name)

    #* FOURTH: clean aux lists 
    # auxiliary lists for 1 FRAME
    color_array.clear()
    depth_array.clear()  
    seg_color_array.clear()
    seg_depth_array.clear()
    pcd_list.clear()


# si pasan los x segundos, sigue
def on_batch_timeout(num_frame):
    with received_frames_lock:
        logger.info(f"Timeout for frame {num_frame} detected")
        if num_frame in received_frames_dict and received_frames_dict[num_frame][0]:
            received_frames_dict[num_frame][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(num_frame)[0]
            threading.Thread(target=process_frames, args=(frame_data_copy, num_frame)).start()

# process_message function extracts all information from the message received then, save frames in batches 
def process_message(msg, total_cameras):
    # get payload
    message = json.loads(msg.payload) # binary data to Python Object ('dictionarity')
    camera_name = message.get('folder_name')
    color_frame_name = message.get('frame_color_name')
    container_name = message.get('container_name')
    K = message.get('K')
    _,_,num_frame = (color_frame_name.split('.')[0]).split('_') # name frame: cameranum_typeframe_numberframe.png
    
    logger.info(f"[TS] Received MSG with frame {num_frame} of the camera {camera_name} and sequence {container_name}")
    
    with received_frames_lock:
        # If the frame number is new, start the batch timer
        if num_frame not in received_frames_dict:  
            received_frames_dict[num_frame] = ([], Timer(batch_timeout, on_batch_timeout, args=(num_frame)))
            received_frames_dict[num_frame][1].start()  # Start the timer

        received_frames_dict[num_frame][0].append(message)  # Store the frame data

        # Check if all cameras for the frame have sent data
        if len(received_frames_dict[num_frame][0]) == total_cameras:
            logger.info(f"Batch full for frame {num_frame}")
            received_frames_dict[num_frame][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(num_frame)[0]  # Get the batch of frames
            threading.Thread(target=process_frames, args=(frame_data_copy, num_frame, container_name, K)).start()
        else:
            # If a new frame arrives, restart the timer
            if received_frames_dict[num_frame][1] is not None:
                received_frames_dict[num_frame][1].cancel()
            received_frames_dict[num_frame] = (received_frames_dict[num_frame][0], Timer(batch_timeout, on_batch_timeout, args=[num_frame]))
            received_frames_dict[num_frame][1].start()

## MQTT funs
def on_message(client, userdata, msg): 
    message = json.loads(msg.payload)
    timestamp = message.get('send_ts')
    total_cameras = message.get('total_cameras')
    total_cameras = int(total_cameras)

    # Timestamp's already received
    if timestamp in processed_timestamps:
        logger.info(f"Timestamp {timestamp} processed. Ignoring...")
        return # it doesnt continue with the rest, waits a new mssg 
    
    processed_timestamps.add(timestamp)
    threading.Thread(target=process_message, args=(msg,total_cameras)).start()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info(f"Connected to HiveMQ Cloud MQTT.")
        client.subscribe(MQTT_TOPIC_CAM)
    else:
        logger.info(f"Connection to MQTT broker failed.")

## MAIN 
def main():
    # Connection to MQTT broker
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME_S, PASSWORD_S)
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)     
    client.loop_forever()

main()