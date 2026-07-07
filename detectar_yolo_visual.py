import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from ultralytics import YOLO
import supervision as sv

def crear_anotadores():
    if hasattr(sv, "BoundingBoxAnnotator"):
        return sv.BoundingBoxAnnotator(), sv.LabelAnnotator()
    if hasattr(sv, "BoxAnnotator"):
        return sv.BoxAnnotator(), sv.LabelAnnotator()
    if hasattr(sv, "RoundBoxAnnotator"):
        return sv.RoundBoxAnnotator(), sv.LabelAnnotator()
    raise AttributeError("No se encontró un anotador compatible en supervision")

# --- Cargar imagen desde URL
url = "https://raw.githubusercontent.com/vlarobbyk/fundamentos-vision-artificial-doctoradoCC/main/images/Catedra-UNESCO-UPS-Imagen-Aula.jpg"
response = requests.get(url)
img_pil = Image.open(BytesIO(response.content)).convert("RGB")
image = np.array(img_pil)

# --- Cargar modelo YOLO
modelo = YOLO("yolo11n.pt")  # Asegúrate de tener el archivo 'yolo11n.pt' en el mismo directorio

# --- Ejecutar predicción
resultados = modelo(image)[0]

# --- Anotar detecciones
detections = sv.Detections.from_ultralytics(resultados)
bounding_box_annotator, label_annotator = crear_anotadores()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# --- Mostrar imágenes original y anotada
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Imagen Original")
axes[0].axis("off")

axes[1].imshow(annotated_image)
axes[1].set_title("YOLOv11 Anotaciones")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# --- Información de cajas detectadas
print("Cajas (xyxy):\n", resultados.boxes.xyxy)
print("Confianza:\n", resultados.boxes.conf)
