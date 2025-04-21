# SERVER
Structure of this directory:<br>
- main_mqtt_handler_cpu.py 
- rest of files: auxiliary scripts (logs, icp algorithm, azure connection, and metrics) 
- /segmentation: directory that contains the trained model and the configurations of the model, these files can be downloaded through OpenMMLab repository (https://github.com/open-mmlab/mmsegmentation) or through this link (https://upm365-my.sharepoint.com/:f:/g/personal/ana_garrido_ruiz_upm_es/EvsnokulLThAgDA6TXpbCd0BygPWUOGvywydhiqa7Cmq-A?e=ndKLzm) the files are: 
    - pspnet_r50-d8_4xb4-80k_ade20k-512x512.py
    - pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth
<br>
<br>
<b>LAUNCH THE SCRIPT under this directory (src_main_cpu): </b><br>
        >> python3 main_mqtt_handler_cpu.py<br>
This script can be launched on CPU, hence, high capacity is needed (the best is GPU)<br>
        >> python3 main_mqtt_handler_gpu.py<br>
By default it chooses the device 0, to choose a define device run: >> CUDA_VISIBLE_DEVICES=yourCUDAnumber python3 main_mqtt_handler_gpu.py<br>