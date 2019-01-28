import cv2
import numpy as np
import csv

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

with open('tabela.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'w', 'h'])

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        invGamma = 0.2
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)

        faces = face_cascade.detectMultiScale(gray)
        for(x, y, w, h) in faces:
            writer.writerow([x, y, w, h])

            nose = nose_cascade.detectMultiScale(gray[y:y+h, x:x+w], 1.3, 5)
            for (x2, y2, w2, h2) in nose:
                cv2.rectangle(img[y:y+h, x:x+w], (x2 + 10, y2 - 20), (x2 + w2 + 10, y2 + 20), (0, 255, 0), 2)

            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            for (x2, y2, w2, h2) in eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(img[y:y+h, x:x+w], (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
        cv2.imshow('imagem', img)
        if cv2.waitKey(30) & 0xff == ord('a'):
            break

cap.release()
cv2.destroyAllWindows()


