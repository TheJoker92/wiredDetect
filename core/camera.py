import cv2
import numpy as np
import imutils


MARKER_REAL_WIDTH = 280
FOCAL_LENGTH = 800

STORED_MARKER = None

CAMERA_ID = "/Users/ADMIN/Downloads/0002.mp4" #0



def add_grid_to_frame(frame, rectangle, grid_size=(100, 100)):
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

    return format_cell(rectangle, rows, cols, row_height, col_width, new_frame)


def format_cell(rectangle, rows, cols, row_height, col_width, new_frame):
    cell_list = []
    cell_num_list = []

    # Add numbers at the center of each cell
    for r in range(rows):
        for c in range(cols):
            cell_center = (c * col_width + col_width // 2, r * row_height + row_height // 2)
            cell_number = r * cols + c + 1
 
            # Calculate the top-left and bottom-right corners of the cell
            cell_top_left = (cell_center[0] - col_width // 2, cell_center[1] + row_height // 2)
            cell_bottom_right = (cell_center[0] + col_width // 2, cell_center[1] - row_height // 2)
                                                        
            if rectangle is not None:
                result = find_cell_overlap(rectangle, cell_bottom_right, cell_top_left, cell_number)

                if result:
                        cell_list.append(result)
                        cell_num_list.append(cell_number)


            cv2.putText(new_frame, str(cell_number), cell_center, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
    for cell in cell_list:
        cell_bottom_right, cell_top_left, cell_number = cell
        # Define the color for the rectangle (you can change this)

        
    

        # Define the rectangle color (BGR format, in this case, red)
        color = (0, 0, 255)

        # Define the thickness of the rectangle border
        thickness = 2  # Use -1 to fill the rectangle

        # Draw the rectangle on the image
        cv2.rectangle(new_frame, cell_bottom_right, cell_top_left, color, thickness)

        # cell_list = []
                
    print(cell_num_list)
    
    return new_frame, cell_num_list

# Function to resize a frame
def scale_frame(frame, scale_percent):
    # Calculate the new dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized_frame


def find_cell_overlap(rectangle, cell_bottom_right, cell_top_left, cell_number):
    
    cell_x_br, cell_y_br = (cell_bottom_right[0]//20, cell_bottom_right[1]//20)
    cell_x_tl, cell_y_tl = (cell_top_left[0]//20, cell_top_left[1]//20)

    rect_x_min = 0
    rect_y_min = 0
    rect_x_max = 0
    rect_y_max = 0

    if rectangle[0][0] < rectangle[1][0]:
        rect_x_min = rectangle[0][0]//20
        rect_x_max = rectangle[1][0]//20
    elif rectangle[0][0] == rectangle[1][0]:
        rect_x_min = rectangle[0][0]//20
        rect_x_max = (rectangle[1][0]//20) + 1
    else:
        rect_x_min = rectangle[1][0]//20
        rect_x_max = rectangle[0][0]//20

    if rectangle[0][1] < rectangle[1][1]:
        rect_y_min = rectangle[0][1]//20
        rect_y_max = rectangle[1][1]//20
    elif rectangle[0][1] == rectangle[1][1]:
        rect_y_min = rectangle[0][1]//20
        rect_y_max = (rectangle[1][1]//20) + 1
    else:
        rect_y_min = rectangle[1][1]//20
        rect_y_max = rectangle[0][1]//20
    

    for x_cell in range(cell_x_tl, cell_x_br):
        for y_cell  in range(cell_y_br, cell_y_tl):
            for x in range(rect_x_min, rect_x_max):
                for y in range(rect_y_min, rect_y_max):
                    if x == x_cell and y == y_cell:
                        return cell_bottom_right, cell_top_left, cell_number            
    

def point_inside_rect(point, top_left, bottom_right):
    return top_left[0] <= point[0] <= bottom_right[0] and top_left[1] <= point[1] <= bottom_right[1]


def create_camera_grid():
    # define a video capture object 
    vid = cv2.VideoCapture(CAMERA_ID) 

    while(True): 
        
    # Capture the video frame 
    # by frame 
        ret, frame = vid.read() 

        if not(ret):
            vid = cv2.VideoCapture(CAMERA_ID) 
            continue

        # frame = cv2.resize(frame, (1200, 720))
        
        
        marker, rectangle = obj_distance(frame)
        
        # grid_frame, num_cells = add_grid_to_frame(frame, rectangle)

        if not(marker == None):
            
            

            inches = (MARKER_REAL_WIDTH * FOCAL_LENGTH) / marker[0][1]

            # if not(inches == None):
            #     draw_inches(inches, grid_frame)

        
        # if rectangle is not None:
        #     cv2.rectangle(frame, rectangle[0], rectangle[1], (0, 255, 0), 2)
        #     cv2.circle(frame, rectangle[0], 20, (0,255,250), 20)

        #     cv2.circle(frame, rectangle[1], 20,  (0,255,255), 20)

        #     cv2.drawContours(frame, [rectangle[2]], -1, (0, 255, 0), 2)


        # grid_frame = cv2.resize(grid_frame, (1200,720))
        # cv2.imshow('frame', grid_frame)

        cv2.imshow('frame', frame)
            
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

def get_factor_cell(num_cell):
    result = 1

        
    if 0 < num_cell <= 10:
        result = 100
    elif 10 < num_cell <= 20:
        result = 7.5
    elif 20 < num_cell <= 30:
        result = 20.4
    elif 30 < num_cell <= 40:
        result = 8
    elif 40 < num_cell <= 50:
        result = 5.4
    elif 50 < num_cell <= 60:
        result = 3.4
    elif 60 < num_cell <= 70:
        result = 3
    elif 70 < num_cell <= 80:
        result = 1.7
    elif 80 < num_cell <= 90:
        result = 1.4
    elif 90 < num_cell <= 100:
        result = 1.1764
    
    print(num_cell)
    
    return result

def distance_to_camera(knownWidth, focalLength, perWidth, num_cells):
	# compute and return the distance from the maker to the camera
    factor = 1

    if len(num_cells) > 0:
        factor = get_factor_cell(num_cells[-1])

    return factor*(focalLength/knownWidth) * perWidth

def obj_distance(frame):
    global STORED_MARKER
    try:

        # Display the resulting frame 
        frame_width, frame_height  = frame.shape[:2]
        frame_clone = frame.copy()

        frame_clone[:frame_width/2 -50, :] = 100
        frame_clone[frame_width/2 +50:, :] = 100
        marker, rectangle = find_marker(frame_clone)
        # print("marker:".format(marker))
        

        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)  
                        
        cv2.drawContours(frame, [box], -1, (255, 155, 0), 2)
        # Calculate the average point (centroid) by taking the mean of the points
        average_point = np.mean(box, axis=0)
        average_point = tuple(map(int, average_point))

        frame_height, frame_width = frame.shape[:2]
        center_of_frame = (frame_width // 2, frame_height // 2)
        center_distance = (average_point[0] - center_of_frame[0], average_point[1] - center_of_frame[1])
        print("{} point of line, distance from center {}".format(average_point, center_distance))

        cv2.circle(frame, average_point, 20, (0,255,250), 10)

        # if STORED_MARKER == None:
        #     STORED_MARKER = marker
        # elif not(STORED_MARKER == marker):
        #     print("SSSSSSS")
        #     raise Exception("other cable detected, not considered") 
        
        return marker, rectangle
    except:
        return None, None

def detect_lines(frame):
    try:
        contours = []
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 220, 255, apertureSize=3)
        
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
            
            # for line in lines:
            #     cv2.line(frame, endpoints(line)[0],endpoints(line)[1],(255,0,0))
            
            # Get endpoints of line1
            (x1, y1), (x2, y2) = endpoints(line1)
            
            # Calculate the length of the sides of the rectangle (half of the width and height)
            width = 10  # Adjust this as needed
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
            # print("REC {}".format(rectangle))
        
            return contours, rectangle
    except Exception as e:
        print (e)
        print("No line")


def draw_inches(inches, frame):
        # draw a bounding box around the image and display it
        cv2.putText(frame, str(2.54*inches) + "[cm]",
            (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (255, 0, 0), 3)
        

