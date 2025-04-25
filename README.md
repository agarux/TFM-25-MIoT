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

### Docker installation (notes for docker installation in AzureVM)
Write line by line (or create a file.sh) in az terminal 

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo usermod -aG docker $USER
```