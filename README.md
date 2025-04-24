# TFM 25 MIoT
Segmentation and interactive visualization of a sequence recorded by depth cameras (these also include rgb information) within an IoT framework.<br>
- Communication protocol: MQTT
- Segmentator: [MMSegmentation](https://github.com/open-mmlab) of OpenMMLab.
- Storage: [Microsoft Azure](https://azure.microsoft.com/en-us/)
  
For more information see README.md files of each subsystem

## Create a virtual environment and install dependencies 
```python
python -m venv name_of_environment 
source name_of_environment/bin/activate
```

then:
```bash
pip install -r requirements.txt
```
