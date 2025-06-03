''' RENAME FILES DIRECTORY BY DIRECTORY only name of files, not structure of all files 
- under the 'src_edgeDevice_sim' run the script >> python3 rename_files 
    - folder: full path of each directory from the src_edgeDevice_sim: '.' to /depth_frames or /color_frames
    - camera: ID of the camera 
    - typeofframe: color or depth
'''

'''

'''
import os

# Function to rename multiple files
def main():
    folder = "./data/cameras/000442922112/depth_frames/"
    camera = "000442922112"
    typeofframe = "depth" # "color"
    extension = "png"
    count=1
    for filename in sorted(os.listdir(folder)):
        print(count)
        dst = f"{camera}_{typeofframe}_f{count:04d}.{extension}"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
        count += 1 
        os.rename(src, dst)
 
if __name__ == '__main__':
    main()