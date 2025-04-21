# Source : tfm mikel: https://github.com/mikeligUPM/tfm_edgecloud_registrator.git
#          Microsoft azure documentation: https://learn.microsoft.com/es-es/azure/storage/blobs/storage-blob-python-get-started?tabs=azure-ad
#          https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d
# V2  Flask is a single-threaded by default 
# Oped 3D must be launched in a separate thread for not blocking purposes 

import os
import threading
import time 
import numpy as np
import open3d as o3d
from flask import Flask, jsonify, request, render_template
from azure.storage.blob import BlobServiceClient
from werkzeug.exceptions import HTTPException

BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=pcdstorageaccount;AccountKey=6TCefowvSStrBGEjazsCxWCWbsDlR80QM9Bq/JwLsm7/u6FYcORAJJKlXx3vWUTSCdEmsSzCCOgn+AStvXatAQ==;EndpointSuffix=core.windows.net"
FRAME_RATE = 1/30

# initialize flask application
app = Flask(__name__)

# Enable WebRTC for Open3D
o3d.visualization.webrtc_server.enable_webrtc()

# initiate a connection to AzureBlobStorage with connection string 
blob_service_client = BlobServiceClient.from_connection_string(conn_str=BLOB_CONNECTION_STRING)
try:
	containers = blob_service_client.list_containers()
except Exception as e:
	print(f"Error listing containers of the Azure Blob Storage. Error: {e}")


# AUX functions 
def download_blob(blob_client, target_blob): # target_blob is the name of the blob: blob.name
	# if the files are already downloaded, dont download them just read them 
	if os.path.isfile(target_blob): 
		return target_blob
	else:
		with open(target_blob, "wb") as download_file:
			download_file.write(blob_client.download_blob().readall())
		return target_blob


def rotate_point_cloud(pc):
    angle_rad = np.radians(180) 
    R = pc.get_rotation_matrix_from_xyz((0, angle_rad, angle_rad))
    pc.rotate(R)
    return pc


def show_blob_o3d(chosen_pointsize, chosen_framerate, chosen_background, point_clouds):
	# Visualization 
	vis = o3d.visualization.Visualizer()
	# create window: determinated size and determinated position
	vis.create_window(window_name='PointCloud visualizer', height=540, width=960, left = 800, top = 300)
	vis.get_render_option().background_color = chosen_background
	vis.get_render_option().point_size = float(chosen_pointsize)
	chosen_framerate = 1/int(chosen_framerate)
	try:
		current_index = 0 
		if point_clouds:
			vis.add_geometry(point_clouds[current_index]) 
		while True:
			vis.remove_geometry(point_clouds[current_index], reset_bounding_box = False)  # false to keep current viewpoint 
			current_index = (current_index + 1) % len(point_clouds) 
			current_point_cloud = point_clouds[current_index]
			vis.add_geometry(current_point_cloud, reset_bounding_box = False)  # false to keep current viewpoint 

			vis.update_geometry(current_point_cloud)
			vis.update_renderer()

			time.sleep(chosen_framerate) 
			if not vis.poll_events():
				break            
	except KeyboardInterrupt:
		print("Closing window...")
	finally:
		vis.destroy_window()


# define route() decorators to bind a function to a URL 
# displays the options in the selectable part. AJAX --> dinamically uploading the options
@app.route("/", methods=['POST', 'GET']) 
def view_cointainer(): 
	return render_template('principal.html') # File saved in src_azurefun/templates/index.html

# list all avaliable cointainers in connection string  
@app.route("/get_containers", methods=['GET'])
def get_containers():
	containers_name = []
	try:
		containers_items = blob_service_client.list_containers()
		containers_name = [container.name for container in containers_items]
	except Exception as e:
		print(e)
	return jsonify (containers_name)

@app.route("/back_to_main", methods=['POST', 'GET'])
def back_to_main():
	return render_template('principal.html')

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    return render_template('error.html', e=e), 500

@app.route("/get_visualization", methods=['POST', 'GET'])
def get_visualization():
	if request.method == 'POST':
		try:
			chosen_container = request.form.get('container_name')
			chosen_pointsize = request.form.get('point_size')
			chosen_framerate = request.form.get('fps_rate')
			chosen_background = request.form.get('color_bkg')
			if chosen_background == 'black':
				chosen_background = [0, 0, 0]
			if chosen_background == 'white':
				chosen_background = [255, 255, 255]
		except Exception as e:
			return "Please provide all parameters."
		
		# download blobs
		container_clients = blob_service_client.get_container_client(chosen_container)
		blob_list = container_clients.list_blobs()
		point_clouds = []
		for blob in blob_list:
				# Get blob client and download the blob content
			blob_client = container_clients.get_blob_client(blob)
			filename = download_blob(blob_client, blob.name)
			# añadir nube de puntos a una lista 
			if filename:
				try:
					point_cloud = o3d.io.read_point_cloud(filename)
					rotate_pc = rotate_point_cloud(point_cloud)
					if not rotate_pc.is_empty():
						point_clouds.append(rotate_pc)
					else:
						print(f"Blob {blob.name} ERROR. Invalid data")
				except Exception as e:
					print(f"ERROR {blob.name}: {e}")
					
		# launch visualizer in a different thread to not blocking the main (Flask) 
		thread = threading.Thread(target=show_blob_o3d, args=(chosen_pointsize, chosen_framerate, chosen_background, point_clouds))
		thread.start()
		return render_template('second.html') 
		

if __name__ == "__main__":
	app.run(debug=False, host='0.0.0.0', port = 5002)