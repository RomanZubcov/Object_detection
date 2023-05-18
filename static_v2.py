import cv2
import cvlib as cv
from cvlib.object_detection import detect_common_objects
import matplotlib.pyplot as plt
import os

# setează calea către folderul cu imaginile
img_folder = '/home/roman/Desktop/proiect'

# iterează prin fiecare imagine din folder și aplică algoritmul de detectare și numărare
for filename in os.listdir(img_folder):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): 
        # încarcă imaginea
        img = cv2.imread(os.path.join(img_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # aplică filtrul bilateral
        filtered_img = cv2.bilateralFilter(img, 9, 75, 75)

        # detectează obiecte comune și filtrează numai dozele de suc
        objects, labels, _ = detect_common_objects(filtered_img, confidence=0.2)
        juice_boxes = [box for i, box in enumerate(objects) if labels[i] == 'bottle']

        # desenează dreptunghiuri delimitatoare cu grosimea de 3 pixeli și adaugă etichete
        for box in juice_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=6)
            cv2.putText(img, 'doza', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 6)

        # numără dozele de suc și afișează imaginea cu dreptunghiuri delimitatoare
        num_juices = len(juice_boxes)
        plt.imshow(img)
        plt.title("Detectare doze de suc și numărare - numărul dozelor de suc: " + str(num_juices))
        plt.axis("off")
        plt.show()

