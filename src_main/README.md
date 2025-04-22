# SERVER
Structure of this directory:

- main_mqtt_handler_cpu.py 
- rest of files: auxiliary scripts (logs, icp algorithm, azure connection, and metrics) 
- /segmentation: directory that contains the trained model and the configurations of the model, these files can be downloaded through OpenMMLab [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repository  or through this [link](https://upm365-my.sharepoint.com/:f:/g/personal/ana_garrido_ruiz_upm_es/EvsnokulLThAgDA6TXpbCd0BygPWUOGvywydhiqa7Cmq-A?e=ndKLzm) the files are: 
    - pspnet_r50-d8_4xb4-80k_ade20k-512x512.py
    - pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth

## Launch the script
Under this directory (src_main_cpu):
```python
python3 main_mqtt_handler_cpu.py
```
This script can be launched on CPU, hence, high capacity is needed (the best is GPU)


```python
python3 main_mqtt_handler_gpu.py
```

By default it chooses the device 0, to choose a define device run:
```python
CUDA_VISIBLE_DEVICES=yourCUDAnumber python3 main_mqtt_handler_gpu.py
```