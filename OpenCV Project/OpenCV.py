import cv2
import numpy as np

# Load the template images
template_1 = cv2.imread('template_1.png', cv2.IMREAD_GRAYSCALE)
template_2 = cv2.imread('template_2.png', cv2.IMREAD_GRAYSCALE)

# Load the video
cap = cv2.VideoCapture('sample.mp4')

# Define the threshold for matching
threshold = 0.3

staff_coordinates = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res_1 = cv2.matchTemplate(gray, template_1, cv2.TM_CCOEFF_NORMED)
    res_2 = cv2.matchTemplate(gray, template_2, cv2.TM_CCOEFF_NORMED)

    max_val_1 = np.max(res_1)
    max_val_2 = np.max(res_2)

    if max_val_1 > threshold or max_val_2 > threshold:
        if max_val_1 > max_val_2:
            template = template_1
        else:
            template = template_2

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_1) if max_val_1 > max_val_2 else cv2.minMaxLoc(res_2)

        staff_coordinates.append((max_loc[0] + template.shape[1] // 2, max_loc[1] + template.shape[0] // 2))

        rect_width = int(template.shape[1] * 0.5)
        rect_height = int(template.shape[0] * 0.5)
        top_left = (max_loc[0] + (template.shape[1] - rect_width) // 2, max_loc[1] + (template.shape[0] - rect_height) // 2)
        bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        
        text = f"X={top_left[0]}, Y={top_left[1]}"
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

