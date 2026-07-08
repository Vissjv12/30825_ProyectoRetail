import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
import os

def crear_anotadores():
    if hasattr(sv, "BoundingBoxAnnotator"):
        return sv.BoundingBoxAnnotator(), sv.LabelAnnotator()
    if hasattr(sv, "BoxAnnotator"):
        return sv.BoxAnnotator(), sv.LabelAnnotator()
    if hasattr(sv, "RoundBoxAnnotator"):
        return sv.RoundBoxAnnotator(), sv.LabelAnnotator()
    raise AttributeError("No se encontró un anotador compatible en supervision")

# --- CONFIGURACIÓN GENERAL
MODELO_PATH = "yolov8n.pt"  # Cambia a tu archivo .pt (yolo11n.pt si lo tienes)
IMAGEN_PATH = "prueba.jpeg"  # Imagen local
SALIDA_PATH = "resultado_yolo.jpeg"  # Imagen de salida

# --- CARGA DEL MODELO
modelo = YOLO(MODELO_PATH)

# --- FUNCIONES DE UTILIDAD
def procesar_con_yolo(frame):
    resultados = modelo(frame)[0]
    detections = sv.Detections.from_ultralytics(resultados)

    bbox_annotator, label_annotator = crear_anotadores()

    anotada = bbox_annotator.annotate(scene=frame.copy(), detections=detections)
    anotada = label_annotator.annotate(scene=anotada, detections=detections)

    return anotada, resultados

# --- MODO 1: DETECCIÓN EN IMAGEN LOCAL
def detectar_en_imagen():
    if not os.path.exists(IMAGEN_PATH):
        print(f"❌ No se encontró la imagen: {IMAGEN_PATH}")
        return

    imagen_bgr = cv2.imread(IMAGEN_PATH)
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    anotada, resultados = procesar_con_yolo(imagen_rgb)

    # Mostrar imagen
    plt.figure(figsize=(12, 6))
    plt.imshow(anotada)
    plt.axis("off")
    plt.title("Detección con YOLO en imagen")
    plt.show()

    # Guardar resultado
    cv2.imwrite(SALIDA_PATH, cv2.cvtColor(anotada, cv2.COLOR_RGB2BGR))
    print(f"✅ Imagen guardada como {SALIDA_PATH}")
    print("Cajas detectadas:\n", resultados.boxes.xyxy.cpu().numpy())
    print("Confianzas:\n", resultados.boxes.conf.cpu().numpy())

# --- MODO 2: DETECCIÓN EN VIDEO DESDE CÁMARA
def detectar_en_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    print("🎥 Presiona ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        anotada, _ = procesar_con_yolo(frame_rgb)
        frame_bgr = cv2.cvtColor(anotada, cv2.COLOR_RGB2BGR)

        cv2.imshow("YOLO detección en tiempo real", frame_bgr)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# --- MENÚ DE EJECUCIÓN
if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Detectar en imagen")
    print("2. Detectar en webcam")
    opcion = input("Opción (1/2): ").strip()

    if opcion == "1":
        detectar_en_imagen()
    elif opcion == "2":
        detectar_en_webcam()
    else:
        print("❌ Opción inválida.")
