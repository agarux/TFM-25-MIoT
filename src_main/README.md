# SERVER
Structure of this directory:

- main_mqtt_handler_cpu.py 
- rest of files: auxiliary scripts (logs, icp algorithm, azure connection, and metrics) 
- /segmentation: directory that contains the trained model and the configurations of the model, these files can be downloaded through OpenMMLab [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) repository  or through this [link](https://upm365-my.sharepoint.com/:f:/g/personal/ana_garrido_ruiz_upm_es/EvsnokulLThAgDA6TXpbCd0BygPWUOGvywydhiqa7Cmq-A?e=ndKLzm) the files are: 
    - pspnet_r50-d8_4xb4-80k_ade20k-512x512.py
    - pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth
## Installation 
### Docker image
```python
docker pull anagarridoupm/tfm25:server
docker run -d --name name_of_container anagarridoupm/tfm25:server
```
  
### Run in local 
Segmentator:
1. Create a virtual environment 
2. Install prerequisites
```python
pip install torch
```
3. Install MMSegmentation
```python
pip install -U openmim
mim install "mmengine==0.10.5"
mim install "mmcv==2.1.0"
pip install "mmsegmentation==1.2.2"
```
4. Install rest of dependencies
```python
pip install -r requirements.txt
```

## Launch the script
Under this directory (src_main):

This script can be launched on CPU platforms, hence, high capacity is needed (the best is GPU)
```python
python3 main_mqtt_handler_cpu.py
```

On GPU platforms. By default it chooses the device 0 
```python
python3 main_mqtt_handler_gpu.py
```

to choose a defined device run:
```python
CUDA_VISIBLE_DEVICES=yourCUDAnumber python3 main_mqtt_handler_gpu.py
```