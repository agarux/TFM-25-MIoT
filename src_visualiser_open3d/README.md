# Visualiser - Open3d
This version allows the user to select a sequence and customizable parameters such as point size, frame rate and color of the background. When the sequence is loaded it is showed in a emerge window. 

## Launch the app
After installing all requirements, run the command:

```bash
python3 app.py 
```

Or build the image and then launch it 

```python
docker run -d -p 5002:5002 --name name_of_container anagarridoupm/tfm25:webapp
```