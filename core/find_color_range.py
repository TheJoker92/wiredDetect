import cv2
import sys
import numpy as np
import imutils

CAMERA_ID = "/Users/ADMIN/Downloads/0002.mp4" #0

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
# frame = cv2.imread("/Users/ADMIN/Desktop/wired/positive/0003.jpeg")
videoCapture = cv2.VideoCapture(CAMERA_ID)
# videoCapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
# videoCapture.set(cv2.CAP_PROP_EXPOSURE, 0.25)

trajectories = []


while(1):

    ret, frame = videoCapture.read()

    if not(ret):
        videoCapture = cv2.VideoCapture(CAMERA_ID)
        continue  

    print(frame.shape)

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
    
    current_frame_trajectories = []

    
    for index in range(len(cnts)):
        print(index)
        
        if index > len(COLOR) - 1:
            print("too elems")
            continue
        c = cnts[index]
        perimeter = cv2.arcLength(c, True)

        distances = []

        x_sum = 0
        y_sum = 0
        for point in c:
            x, y = point[0]
            x_sum = x_sum + x
            y_sum = y_sum + y

            dist = np.linalg.norm(np.array([x, y]) - np.array(reference_point))
            distances.append(dist)
    
        x_avg = x_sum/len(c)
        y_avg = y_sum/len(c)

        center = (x_avg,y_avg)

        if perimeter < w + h:
            print("low perimenters")
            continue

        border = cv2.drawContours(output, [c], -1, COLOR[index][1], 2)
        if len(trajectories) == 0:
            # Initialize trajectories if it's the first frame
            trajectories = [[center] for center in current_frame_trajectories]
        else:
            # Match contours using simple proximity method
            for trajectory in trajectories:
                if len(trajectory) > 0:
                    last_position = trajectory[-1]
                    min_distance = float('inf')
                    best_match = None
                    
                    for i, center in enumerate(current_frame_trajectories):
                        if(last_position == None or center == None):
                            continue

                        distance = np.linalg.norm(np.array(last_position) - np.array(center))
                        if distance < min_distance:
                            min_distance = distance
                            best_match = i
                    
                    if best_match is not None and min_distance < 50:  # Threshold for matching
                        trajectory.append(current_frame_trajectories[best_match])
                        current_frame_trajectories[best_match] = None
            
            # Add any new contours that did not match existing trajectories
            for center in current_frame_trajectories:
                if center is not None:
                    trajectories.append([center])
        
        # Draw the trajectories
        for trajectory in trajectories:
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                cv2.line(output, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        

        

        current_frame_trajectories.append((x_avg,y_avg))


        cv2.circle(output, (x_avg,y_avg), radius=30, color=COLOR[index][1], thickness=-1)

        
        distance_respect_to_center = min(distances)
        distance_cam = calculate_distance(distance_respect_to_center, 12)

        point1 = (510, 150)
        point2 = (410, 150)

        # Distanza reale tra i due punti noti (in metri o centimetri)
        real_distance = 4.0  # ad esempio 1 metro

        # Calcola la distanza in pixel tra i due punti
        pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

        # Calcola il fattore di scala (distanza reale per pixel)
        scale_factor = real_distance / pixel_distance

        
        cable_point1 = (x_avg, y_avg)

        # Distanza in pixel del cavo
        cable_pixel_distance = np.sqrt((cable_point1[0] - reference_point[0]) ** 2 + (cable_point1[1] - reference_point[1]) ** 2)

        # # Distanza reale del cavo
        # cable_real_distance = cable_pixel_distance * scale_fact
        print("Distanza minima dal punto di riferimento: {} colore {} pixel {} cm distance to center ".format(COLOR[index][0], distance_respect_to_center, cable_pixel_distance))

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()