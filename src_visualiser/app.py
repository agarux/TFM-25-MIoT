import os
from flask import Flask, send_from_directory, jsonify, request
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Azure connection 
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=pcdstorageaccount;AccountKey=6TCefowvSStrBGEjazsCxWCWbsDlR80QM9Bq/JwLsm7/u6FYcORAJJKlXx3vWUTSCdEmsSzCCOgn+AStvXatAQ==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
current_folder = '' 

# AUX: download file from the azure portal if it doesnt exist in local, return the new/exist file
def download_blob (blob_client, file_path, chosen_container):
    file_path = os.path.join(chosen_container, file_path)
    if os.path.isfile(file_path):
        return file_path
    with open(file_path, "wb") as f:
        download_file = blob_client.download_blob()
        download_file.readinto(f)
    return file_path

@app.route('/download_ply/<filename>')
def download_ply(filename):
    file_path = os.path.join(current_folder, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(directory=current_folder, path=filename)

@app.route("/get_framesURL")
def get_framesURL():
    urls = sorted(os.listdir(current_folder))
    return jsonify(urls)

@app.route("/get_visualization", methods=['POST'])
def get_visualization():
    if request.method == 'POST':
        chosen_container = request.form.get('container_name')
        if not os.path.exists(chosen_container):
            os.makedirs(chosen_container)
    
    global current_folder # asigna el nombre de la secuencia elegida a una variable global 
    current_folder = chosen_container # asi es visible por el download_ply (@app.route)
    
    container_client = blob_service_client.get_container_client(chosen_container)
    blob_list = container_client.list_blobs()
    frame_names = []
    for blob in blob_list:
        blob_client = container_client.get_blob_client(blob)
        pathfiledown = download_blob(blob_client, blob.name, chosen_container)
        frame_names.append(os.path.basename(pathfiledown))

    return jsonify({'redirect_url': './static/second.html'})

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

# Ruta para servir la página MAIN HTML 
@app.route('/')
def index():
    return send_from_directory('static', 'layout.html')

# MAIN
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5002)