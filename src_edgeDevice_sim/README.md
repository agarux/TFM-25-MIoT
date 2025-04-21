# EDGE DEVICES
Structure of this directory:<br>
- cam_params.json: extrinsic and instrinsic parameters of the kinect cameras (8)
- cameras_sim.py: main
- logger_config.py: used to control the functionallity of this system 
- requirements.txt: dependencies
- data/basedirectory: directory that contains all frames 
- rename_files.py
<br>
<br>
<b> LAUNCH THE SCRIPT under this directory (src_edgeDevice_sim): </b><br>
>> python3 cameras_sim.py<br>

- As it is a simulation the directoty that contains all frames must be inside 'src_edgeDevice_sim' and must have this structure<br>
        /data<br>
        /data/cameras<br>
        /data/cameras/000442922112<br>    
        /data/cameras/000442922112/color<br>
        /data/cameras/000442922112/depth<br>
        /data/cameras/otra camara<br>    
        /data/cameras/otra camara/color<br>
        /data/cameras/otra camara/depth<br>
- The name of the files must have this structure: cameraID_color/depth_fnumberofframe.png, example: 000442922112_color_f0004.png <br>
The rename_files.py is used to rename all the files to have the requied name's structure, user must define all variables of the script. It is not used for reordering the directories (structure of the first bullet point)
- To change the data: change the path at line 198: basedirectory ./data/yourdata<br>
- Downsamplig is needed: set factor at lines 115 and 119<br>
