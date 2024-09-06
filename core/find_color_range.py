import cv2
import sys
import numpy as np
import imutils

CAMERA_ID = 0

COLOR = [
    ["Rosso", (255, 0, 0)],
    ["Verde", (0, 255, 0)],
    ["Blu", (0, 0, 255)],
    ["Giallo", (255, 255, 0)],
    ["Ciano", (0, 255, 255)],
    ["Magenta", (255, 0, 255)],
    ["Arancione", (255, 165, 0)],
    ["Viola", (128, 0, 128)],
    ["Lime", (0, 255, 128)],
    ["Rosa", (255, 192, 203)],
    ["Marrone", (165, 42, 42)],
    ["Grigio Chiaro", (211, 211, 211)],
    ["Oro", (255, 215, 0)],
    ["Argento", (192, 192, 192)]
]

FOCAL_LENGTH = 800

def calculate_distance(real_width, actual_width):
    global FOCAL_LENGTH
    return (actual_width * FOCAL_LENGTH) / real_width



def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# /Users/ADMIN/Desktop/wired/positive/0007.jpeg
# /Users/ADMIN/Downloads/09780b72-a4d3-4b81-a267-f9c2c160f7f5.jpeg
frame = cv2.imread("/Users/ADMIN/Desktop/wired/positive/0007.jpeg")
# videoCapture = cv2.VideoCapture(CAMERA_ID)
# videoCapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
# videoCapture.set(cv2.CAP_PROP_EXPOSURE, 0.25)


while(1):

    # ret, frame = videoCapture.read()

    # h, s, v = cv2.split(frame)
    # v += 50
    # final_hsv = cv2.merge((h, s, v))

    # frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame,frame, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    h, w = frame.shape[:2]
    reference_point = (w // 2, h // 2)  # Centro dell'immagine
    
    
    for index in range(len(cnts)):
        print(index)
        
        if index > len(COLOR) - 1:
            continue
        c = cnts[index]
        perimeter = cv2.arcLength(c, True)

        if perimeter < w + h:
            continue
        border = cv2.drawContours(output, [c], -1, COLOR[index][1], 2)
        distances = []
        for point in c:
            x, y = point[0]
            dist = np.linalg.norm(np.array([x, y]) - np.array(reference_point))
            distances.append(dist)
    
        distance_respect_to_center = min(distances)
        distance_cam = calculate_distance(distance_respect_to_center, 12)
        print("Distanza minima dal punto di riferimento: {} pixel distance to center of frame {} colore".format(distance_respect_to_center, COLOR[index][0]))

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()