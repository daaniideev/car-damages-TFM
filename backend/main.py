from ultralytics import YOLO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Importar la extensión CORS
import cv2
from functions import get_model_route, getDamages, createTxtFile, is_video_file
import os
import shutil
import json
app = Flask(__name__)

CORS(app)

# Definir la ruta para obtener las partes del coche
@app.route('/api/get-car-damages', methods=['GET'])

def getCarDamages():
    # Obtener el parámetro 'video_name' de la cadena de consulta
    video_name = request.args.get('video_name', '-')  # El valor predeterminado es '-'

    damages = getDamages(car_parts_route, car_damgs_route, video_name)
    response_data = {
        "message": damages,
        "status": "success"
    }

    return jsonify(response_data), 200

@app.route('/api/web-images/<filename>')
def serve_image(filename):
    return send_from_directory('web-images', filename)


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"message": "No file part", "status": "error"}), 400

    file = request.files['file']
    filename = request.form.get('filename')  # Obtener el nombre del archivo desde el formulario
    if file.filename == '':
        return jsonify({"message": "No selected file", "status": "error"}), 400

    # Verificar que el archivo sea un video 
    if is_video_file(file):
        file.save(os.path.join(filename.split('/')[0], filename.split('/')[1]))  # Guardar el archivo con el nombre proporcionado

        return jsonify({"message": "File successfully uploaded", "status": "success"}), 200
    else:
        return jsonify({"message": "Invalid file format. Only video is allowed.", "status": "error"}), 400


@app.route('/api/report-errors', methods=['POST'])
def report_errors():
    damages_array = request.form.get('damagesArray').split(',')
    original_image_route = request.form.get('imageRoute')
    damage_coords = request.form.get('damageCoord').replace(',', ' ')

    txt_route = 'retrained-dataset/labels/' +  original_image_route.replace('web-images/', '').split('.')[0] + '.txt'
    directory_aux = txt_route.split('/')

    directory = '/'.join(directory_aux[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory) 

    # Guardo el txt en el dataset reentrenado
    createTxtFile(damages_array, txt_route, damage_coords)
    # Guardo la imagen en el dataset reentrenado

    return jsonify(''), 200


if __name__ == '__main__':

    prefix_car_parts = 'models/car_parts/'
    prefix_car_damgs = 'models/car_damages/'
    prefix_videos = 'videos/'

    # Obtener las rutas de los modelos
    car_damgs_route = get_model_route(prefix_car_damgs)
    car_parts_route = get_model_route(prefix_car_parts)

    # Ejecutar el servidor en el puerto 5000
    app.run(host='0.0.0.0', port=5000)
