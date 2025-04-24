# EDGE DEVICES
Structure of this directory:
- cam_params.json: extrinsic and instrinsic parameters of the kinect cameras (8)
- cameras_sim.py: main
- logger_config.py: used to control the functionallity of this system 
- requirements.txt: dependencies
- data/basedirectory: directory that contains all frames 
- rename_files.py
- 
## Launch the script 
Under this directory (src_edgeDevice_sim): 
```python
python3 cameras_sim.py
```
- As it is a simulation the directoty that contains all frames must be inside 'src_edgeDevice_sim' and must have this structure

        /data
        /data/cameras
        /data/cameras/000442922112   
        /data/cameras/000442922112/color
        /data/cameras/000442922112/depth
        /data/cameras/otra camara  
        /data/cameras/otra camara/color
        /data/cameras/otra camara/depth

- The name of the files must have this structure: 
cameraID_color/depth_fnumberofframe.png, example: 000442922112_color_f0004.png

The rename_files.py is used to rename all the files to have the requied name's structure, user must define all variables of the script. It is not used for reordering the directories (structure of the first bullet point)
- To change the data: change the path at line 198: basedirectory ./data/yourdata
- Downsamplig is needed: set factor at lines 115 and 119

## Docker 
- Run the docker with -i (interaction) to enter name of sequence and number of cameras used 
- A downsamplig factor is set (1/20) by default
- There is data in the container, to upload new data:
    - docker cp ... or ...
    - build again the image 

```python
docker pull anagarridoupm/tfm25:edge
docker run -d -i --name name_of_container anagarridoupm/tfm25:edge
```