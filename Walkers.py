# Importar la biblioteca OpenCV
import cv2


# Crear nuestro clasificador de cuerpos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicializar la captura de video para nuestro archivo de video
cap = cv2.VideoCapture('walking.avi')

# Comenzar el bucle una vez que el video esté cargado exitosamente
while True:
    
    # Leer el primer cuadro
    ret, frame = cap.read()

    # Convertir cada cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasar el cuadro a nuestro clasificador de cuerpos
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extraer las cajas envolventes para cualquier cuerpo identificado
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color=frame[y:y+h,x:x+w]
        cv2.imwrite("Proyecto106.jpg",roi_color)

    # Mostrar el cuadro resultante 
    cv2.imshow("PRO C106 DETECTAR AL PEATON", frame)

    # Salir de la ventana a través de la barra espaciadora
    if cv2.waitKey(1) == 32: #32 es la barra espaciadora
        break

# Después del bucle, liberar el objeto capturado
cap.release()

# Destruir todas las ventanas
cv2.destroyAllWindows()
