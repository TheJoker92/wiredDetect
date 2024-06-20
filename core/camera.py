import cv2
import numpy as np
import imutils

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 12.0


def add_grid_to_frame(frame, grid_size=(10, 10)):
    """
    Draws a grid on the frame and returns the new frame.

    Args:
        frame (ndarray): The video frame on which to draw the grid.
        grid_size (tuple): Number of rows and columns in the grid.
    
    Returns:
        ndarray: The new frame with the grid drawn.
    """
    new_frame = frame.copy()
    frame_height, frame_width = new_frame.shape[:2]
    rows, cols = grid_size
    row_height = frame_height // rows
    col_width = frame_width // cols

    # Draw horizontal lines
    for r in range(1, rows):
        y = r * row_height
        cv2.line(new_frame, (0, y), (frame_width, y), (0, 255, 0), 1)

    # Draw vertical lines
    for c in range(1, cols):
        x = c * col_width
        cv2.line(new_frame, (x, 0), (x, frame_height), (0, 255, 0), 1)

    add_number_cell(rows, cols, row_height, col_width, new_frame)

    return new_frame


def add_number_cell(rows, cols, row_height, col_width, new_frame):
    # Add numbers at the center of each cell
    for r in range(rows):
        for c in range(cols):
            cell_center = (c * col_width + col_width // 2, r * row_height + row_height // 2)
            cell_number = r * cols + c + 1
            cv2.putText(new_frame, str(cell_number), cell_center, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)


def create_camera_grid():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 

    while(True): 
        
    # Capture the video frame 
    # by frame 
        ret, frame = vid.read() 
        
        
        # Display the resulting frame 
        inches, rectangle = obj_distance(frame)
        print(inches)
        grid_frame = add_grid_to_frame(frame)

        if not(inches == None):
            draw_inches(inches, grid_frame)

        
        if not(rectangle == None):
            cv2.rectangle(frame, rectangle[0], rectangle[1], (0, 255, 0), 2)
            cv2.circle(frame, rectangle[0], 20, (0,255,250), 20)

            cv2.circle(frame, rectangle[1], 20,  (0,255,255), 20)

            cv2.drawContours(frame, [rectangle[2]], -1, (0, 255, 0), 2)



        cv2.imshow('frame', grid_frame)

            
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def find_marker(frame):
    try:
        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts, rectangle = detect_lines(frame)

        # cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)
        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c), rectangle
    except:
        pass

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def obj_distance(frame):
    try:
        marker, rectangle = find_marker(frame)
        focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(frame, [box], -1, (0, 0, 255), 2)
        
        return inches, rectangle
    except:
        return None, None

def detect_lines(frame):
    try:
        contours = []
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 220, 250, apertureSize=3)
        
        # This returns an array of r and theta values
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        if lines is not None and len(lines) > 0:
            lines = np.squeeze(lines)
    
            # Filter and find lines (you may want to adjust this part)
            line1 = lines[0]  # Selecting the first line, you may need to adjust this
            
            # Function to find endpoints of the line
            def endpoints(line):
                rho, theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 400 * (-b))
                y1 = int(y0 + 400 * (a))
                x2 = int(x0 - 1200 * (-b))
                y2 = int(y0 - 1200 * (a))
                return (x1, y1), (x2, y2)
            
            # Get endpoints of line1
            (x1, y1), (x2, y2) = endpoints(line1)
            
            # Calculate the length of the sides of the rectangle (half of the width and height)
            width = 100  # Adjust this as needed
            height = 50  # Adjust this as needed
            
            # Calculate points for the rectangle
            dx = x2 - x1
            dy = y2 - y1
            magnitude = np.sqrt(dx**2 + dy**2)
            ux = 0.1*dx / magnitude  # Unit vector along the line direction
            uy = 0.1*dy / magnitude
            
            # Points of the rectangle
            p1 = (int(x1 + width * uy), int(y1 - width * ux))
            p2 = (int(x1 - width * uy), int(y1 + width * ux))
            p3 = (int(x2 - width * uy), int(y2 + width * ux))
            p4 = (int(x2 + width * uy), int(y2 - width * ux))

                            
            contour = np.array([tuple(p1), tuple(p1), tuple(p2), tuple(p3), tuple(p4)], dtype=np.int32)
            contours.append(contour)

            rectangle = [tuple(p2), tuple(p3), contour]
            print("REC {}".format(rectangle))
        
            return contours, rectangle
    except Exception as e:
        print (e)
        print("No line")


def draw_inches(inches, frame):
        # draw a bounding box around the image and display it
        cv2.putText(frame, str(inches) + "[cm]",
            (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (255, 0, 0), 3)
        