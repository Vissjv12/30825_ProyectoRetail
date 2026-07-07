import cv2
from ultralytics import YOLO

# Cargar modelo YOLOv8 preentrenado
modelo = YOLO("yolov8n.pt")  # Usa yolov8n, yolov8s, etc. según tu GPU

# Iniciar la cámara (o usa ruta a un video)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar para mejorar rendimiento (opcional)
    frame_redim = cv2.resize(frame, (1042, 880))

    # Realizar detección
    resultados = modelo.predict(frame_redim, conf=0.4)

    # Dibujar resultados
    for resultado in resultados:
        cajas = resultado.boxes.xyxy.cpu().numpy()
        clases = resultado.boxes.cls.cpu().numpy()
        confs = resultado.boxes.conf.cpu().numpy()
        nombres = resultado.names

        for (x1, y1, x2, y2), clase_id, conf in zip(cajas, clases, confs):
            etiqueta = f"{nombres[int(clase_id)]} {conf:.2f}"
            color = (0, 255, 0) if nombres[int(clase_id)] == "person" else (255, 0, 0)

            # Dibujar rectángulo y etiqueta
            cv2.rectangle(frame_redim, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame_redim, etiqueta, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar resultado
    cv2.imshow("Detección Facial y de Figuras con YOLOv8", frame_redim)

    # Salir con 'Esc'
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
