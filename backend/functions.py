import os
import cv2
import numpy as np

from ultralytics import YOLO


def convertVideoToFramesArray(video_path):
	frames_array = []
	cap = cv2.VideoCapture('videos/' + video_path)
	if not cap.isOpened():
			print("Error al abrir el archivo de video")
	else:
			total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

			for frame_index in range(total_frames):
					ret, frame = cap.read()
					if not ret:
							print(f"No se pudo leer el frame {frame_index}")
							continue

					frames_array.append(frame)

			cap.release()
	return frames_array

def createTxtFile(damagesArray, fileName, coordenadas):
    damages_dict = {
        "broken lamp": 0,
        "glass shatter": 1,
        "scratch": 2,
        "dent": 3,
        "tire flat": 4,
    }

    if os.path.exists(fileName):
        os.remove(fileName)

    content = ''

    if 'no-damage' not in damagesArray:
        for element in damagesArray:
            damageNum = damages_dict[element]
            print("coordenadas")
            print(coordenadas)
            content = content + str(damageNum) + ' ' + coordenadas + '\n'

    with open(fileName, "w") as file:
        file.write(content)

def is_damage_possible(nombre_de_clase_2, numero_de_clase):
    diccionario_combinaciones_posibles = {
        0: ['scratch', 'dent'],
        1: ['tire flat'],
        2: ['glass shatter'],
        3: ['scratch', 'dent'],
        4: ['scratch', 'dent'],
        5: ['scratch', 'dent'],
        6: [],
        7: ['glass shatter'],
        8: ['glass shatter'],
        9: ['scratch', 'dent'],
        10: ['broken lamp'],
        11: ['tire flat'],
        12: ['glass shatter'],
        13: ['scratch', 'dent'],
        14: ['scratch', 'car_damages_confdent'],
        15: ['broken lamp'],
        16: [],
        17: ['scratch', 'dent'],
        18: ['scratch', 'dent'],
        19: ['scratch', 'dent'],
        20: ['scratch', 'dent'],
    }
    if nombre_de_clase_2 in diccionario_combinaciones_posibles[numero_de_clase]:
        return True
    else:
        return False
    
def misma_imagen(imagenes, imagen_aux):
	for element in imagenes:
			parecido = calcular_similitud_histograma(element, imagen_aux)
			if parecido >= 0.7:
					return True
	return False   

def guardar_imagen(ruta_para_guardar, imagen_aux):
    # Guardo la imagen:
    cv2.imwrite(ruta_para_guardar, imagen_aux)
    
def obtener_imagen(parte_de_la_imagen_xyxyn, imagen_original):
    if len(parte_de_la_imagen_xyxyn) != 4 or any(not isinstance(coord, (float, int)) for coord in parte_de_la_imagen_xyxyn):
        raise ValueError("Las coordenadas deben ser una lista de cuatro valores numéricos [x_min, y_min, x_max, y_max].")
    
    x_min, y_min, x_max, y_max = parte_de_la_imagen_xyxyn
    h, w, _ = imagen_original.shape
    
    x_min = max(0, int(x_min * w))
    y_min = max(0, int(y_min * h))
    x_max = min(w, int(x_max * w))
    y_max = min(h, int(y_max * h))
    
    imagen_recortada = imagen_original[y_min:y_max, x_min:x_max]
    
    if imagen_recortada.size == 0:
        print(f"Recorte vacío: coordenadas fuera de los límites de la imagen.")
    
    return imagen_recortada

def calcular_similitud_histograma(imagen1, imagen2):
    # Convertir las imágenes a escala de grises
    imagen1_gray = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    imagen2_gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Calcular los histogramas de las dos imágenes
    hist1 = cv2.calcHist([imagen1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([imagen2_gray], [0], None, [256], [0, 256])
    
    # Normalizar los histogramas
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calcular la correlación entre los histogramas
    similitud = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Convertir la similitud en porcentaje
    similitud_porcentaje = similitud
    
    return similitud_porcentaje	


def anadir_info_del_dano(ruta_para_guardar, conf_damage, nombre_de_clase_1, nombre_de_clase_2, damages, coord):
    damages_traduction = {
	"broken lamp": "faro roto",
		"glass shatter": "cristal roto",
		"scratch": "rallada",
		"dent": "bolladura",
		"tire flat": "rueda pinchada"
}
    
    info = {
        'car_damage_route' : ruta_para_guardar,
        'car_damage_conf' : conf_damage,
        'car_part_name' : nombre_de_clase_1,
        'car_damage' : damages_traduction[nombre_de_clase_2],
        'car_damage_coord' : coord,
    }

    damages.append(info)
    return damages

def is_confidence_sufficient(nombre_de_clase_2, conf_damage):

    mult = 0.5
    
    car_damages_config = {
            "broken lamp": 0.94 * mult,
            "glass shatter": 0.88 * mult,
            "scratch": 0.88 * mult,
            "dent": 0.81 * mult,
            "tire flat": 0.95 * mult
        }

    if conf_damage >= car_damages_config[nombre_de_clase_2]:
        return True
    else:
        return False
    
def is_video_file(file):
	video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.mpeg', '.mpg')
	return file.filename.lower().endswith(video_extensions)


def draw_red_square(image_array, coords):
    image_with_square = image_array.copy()

    height, width, _ = image_with_square.shape

    x_min = int(coords[0] * width)
    y_min = int(coords[1] * height)
    x_max = int(coords[2] * width)
    y_max = int(coords[3] * height)

    color = (0, 0, 255)
    thickness = 2

    cv2.rectangle(image_with_square, (x_min, y_min), (x_max, y_max), color, thickness)

    return image_with_square

def get_last_model(file_list, prefix):
    # Verifica si hay archivos en la lista
    if not file_list:
        raise FileNotFoundError("No se encontraron archivos en el directorio especificado.")
    
    # Ordena la lista de archivos por fecha de modificaciÃ³n
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(prefix, x)), reverse=True)
    return file_list[0]

def get_model_route(prefix):
    # Obtener la lista de archivos en el directorio especificado
    archivos = [f for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f))]
    
    # Obtener el Ãºltimo modelo (segÃºn el criterio deseado)
    last_model_name = get_last_model(archivos, prefix)

    # Retornar la ruta completa al Ãºltimo modelo
    return os.path.join(prefix, last_model_name)

	
def getDamages(car_parts_model_path, car_damgs_model_path, video_path):
    num_of_image = 0
    damages = []
    imagenes = []
    frames_array = []
    damages_arr = []
    
    # Paso los videos a un array de frames:
    frames_array = convertVideoToFramesArray(video_path)
    
    # Itero sobre todos los frames del video:
    for frame_index, frame in enumerate(frames_array):
        # Predigo las partes del vehículo:
        results = YOLO(car_parts_model_path).predict(source=frame, save=False, show=False, device='cpu', conf=0.9, verbose=False)
        
        for element in results[0].boxes:
            parte_de_la_imagen = obtener_imagen(element.xyxyn.tolist()[0], frame)
            numero_de_clase = int(element.cls.tolist()[0])
            nombre_de_clase_1 = results[0].names[numero_de_clase]
    
            # Predigo los daños del vehículo:
            results_2 = YOLO(car_damgs_model_path).predict(source=parte_de_la_imagen, save=False, show=False, device='cpu', verbose=False)
    
            # Itero sobre los resultados de las predicciones de los daños del vehículo detectados:
            for element_2 in results_2[0].boxes:
                numero_de_clase_2 = int(element_2.cls.tolist()[0])
                parte_de_la_imagen_2 = obtener_imagen(element_2.xyxyn.tolist()[0], parte_de_la_imagen)
                nombre_de_clase_2 = results_2[0].names[numero_de_clase_2]
                conf_damage = float(element_2.conf.tolist()[0])
            
                if is_damage_possible(nombre_de_clase_2, numero_de_clase) and is_confidence_sufficient(nombre_de_clase_2, conf_damage):
                            
                        damages_arr.append([frame_index, numero_de_clase, numero_de_clase_2])
                        num_of_image = num_of_image + 1
                        imagen_aux = cv2.resize(parte_de_la_imagen, (256, 256))
                        if len(imagenes) == 0 or not misma_imagen(imagenes, imagen_aux):
        
                            # Guardo la imagen en la carpeta del dataset:
                            os.makedirs('retrained-dataset/images/', exist_ok=True)
                            ruta_imagen_ds = 'retrained-dataset/images/'  + video_path.split('/')[-1].split('.')[0] + '_' + str(num_of_image) + '.jpg'
                            guardar_imagen(ruta_imagen_ds, imagen_aux)

                            # Guardo la label en la carpeta del dataset:
                            os.makedirs('retrained-dataset/labels/', exist_ok=True)
                            ruta_label_ds = 'retrained-dataset/labels/'  + video_path.split('/')[-1].split('.')[0] + '_' + str(num_of_image) + '.txt'
                            createTxtFile([nombre_de_clase_2], ruta_label_ds, ' , '.join(map(str, element_2.xyxyn.tolist()[0])))

                            # Guardo la imagen para poder verla desde la web:
                            os.makedirs('web-images/', exist_ok=True)
                            imagen_aux_square = draw_red_square(imagen_aux, element_2.xyxyn.tolist()[0])
                            ruta_imagen_web = 'web-images/' + video_path.split('/')[-1].split('.')[0] + '_' + str(num_of_image) + '.jpg'
                            guardar_imagen(ruta_imagen_web, imagen_aux_square)

                            # Guardo la imagen en el array:
                            imagenes.append(imagen_aux_square)
            
                            damages = anadir_info_del_dano(ruta_imagen_web, conf_damage, nombre_de_clase_1, nombre_de_clase_2, damages, element_2.xyxyn.tolist()[0])
    
    return damages