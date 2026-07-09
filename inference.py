from pathlib import Path
import cv2 as cv
from ultralytics import YOLO


base_dir = Path(__file__).resolve().parent
model_path = base_dir / "best.pt"

#Cambiar el nombre del video por Minerales.mp4 o Minerales2.mp4 según el video que se quiera usar
#El primero es una situación mas casual mientras que en el segundo es de noche, con algunos minerales en cuevas y con menos luz.
video_path = base_dir / "Minerales.mp4"

if not model_path.exists():
    raise FileNotFoundError(f"No se encontró el modelo entrenado en: {model_path}")

if not video_path.exists():
    raise FileNotFoundError(f"No se encontró el video en: {video_path}")

model = YOLO(str(model_path))
cap = cv.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"No se pudo abrir el video: {video_path}")
else:
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video", 1720, 1060)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o no se pudo leer el frame")
            break

        #opté por elegir un umbral de confianza de 0.7 para filtrar detecciones menos confiables
        #según la curva Precision-Confidence del modelo, un umbral de 0.705 descarta la mayoría de las 
        #detecciones falsas positivas, manteniendo una buena precisión en las detecciones verdaderas.
        results = model(frame, conf=0.7, verbose=False)
        #Primero habia optado por usar un umbral de 0.45 basado en la curva F1-Confidence del modelo, pero al analizar los resultados,
        #observé que habian detecciones falsas positivas, lo que afectaba la precisión general del modelo. Por lo tanto, decidí aumentar el umbral a 0.7
        #para mejorar la confiabilidad de las detecciones, aunque esto pueda reducir la cantidad de objetos detectados.



        frame_a_mostrar = frame.copy()
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                color = (0, 255, 0)
                cv.rectangle(frame_a_mostrar, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    frame_a_mostrar,
                    label,
                    (x1, max(10, y1 - 5)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv.LINE_AA,
                )

        cv.imshow("Video", frame_a_mostrar)

        # Salir con ESC
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

    #El modelo presenta ciertas limitaciones, 
    #como la dificultad para detectar objetos pequeños, o parcialmente ocultos, y en zonas alejadas del centro de la camara del juagador.
    #aun así, el modelo logra identificar correctamente los minerales en el video, con buen valor de seguridad,
    #demostrando su eficacia en la tarea de detección de objetos.

    #La mejor clase sigue siendo la piedra solar, mientras que el platino y cuarzo estan parejas para decidir cual es la peor 
    #(Tampoco son malas clases pero si estan un poco por debajo de la piedra solar),
    #el modelo las logra detectar con un buen nivel de confianza.

    #Al usar el umbral de confianza de 0.7, el modelo dejó de mostrar los falsos positivos 
    #que sí aparecían con el umbral de 0.45, lo que confirma que el ajuste mejoró la confiabilidad.