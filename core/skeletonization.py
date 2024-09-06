import cv2
import numpy as np

def skeletonize(image):
    # Converte l'immagine in binario se necessario
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Inizializza lo scheletro
    skeleton = np.zeros(binary.shape, np.uint8)

    # Struttura kernel per le operazioni morfologiche
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Ciclo per scheletrizzare
    while True:
        # Erosione
        eroded = cv2.erode(binary, kernel)
        # Dilatazione dell'erosione per ottenere la parte interna
        temp = cv2.dilate(eroded, kernel)
        # Sottrazione dell'immagine erosa dall'immagine originale
        temp = cv2.subtract(binary, temp)
        # Combina con lo scheletro
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()

        # Condizione di uscita: nessun pixel bianco
        if cv2.countNonZero(binary) == 0:
            break

    return skeleton

import cv2
import numpy as np

def calculate_thickness(skeleton, contours):
    thickness_map = np.zeros(skeleton.shape, dtype=np.float32)

    # Per ogni punto del "skeleton", trova la distanza minima dal contorno
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] == 255:  # Controlla se il punto fa parte dello scheletro
                min_dist = float('inf')
                for contour in contours:
                    for point in contour:
                        contour_point = point[0][:2] 
                        dist = np.linalg.norm(np.array([x, y]) - contour_point)
                        if dist < min_dist:
                            min_dist = dist
                thickness_map[y, x] = min_dist * 2
    return thickness_map

# Esempio di utilizzo con i contorni e lo scheletro:
# Carica l'immagine in scala di grigi
image = cv2.imread('/Users/ADMIN/Downloads/09780b72-a4d3-4b81-a267-f9c2c160f7f5.jpeg', cv2.IMREAD_GRAYSCALE)

# Esegui il rilevamento dei contorni
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

skeleton = skeletonize(binary)

# Calcola lo spessore per ogni elemento scheletrizzato
thickness_map = calculate_thickness(skeleton, contours)

# Visualizza lo scheletro e la mappa dello spessore
cv2.imshow('Scheletro', skeleton)
cv2.imshow('Mappa dello Spessore', thickness_map / np.max(thickness_map))  # Normalizza per la visualizzazione
cv2.waitKey(0)
cv2.destroyAllWindows()