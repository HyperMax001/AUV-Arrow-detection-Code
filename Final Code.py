import cv2
import numpy as np
import math

def empty(a):
    pass

def line_equation_end_finder(points1, points2, img_end_x):
    x, y = points1
    x3, y3 = points2 
    slope = (y3 - y) / (x3 - x)
    img_end_y = slope * (img_end_x - x) + y
    img_start_y = slope * (0 - x) + y
    return (int(img_end_y), int(img_start_y), float(slope))

img1 = cv2.imread("Picture1.jpg")
img2 = cv2.imread("Picture2.jpg")
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

lower = np.array([9, 0, 0])
upper = np.array([172, 255, 255])

mask1 = cv2.inRange(img1_hsv, lower, upper)
mask2 = cv2.inRange(img2_hsv, lower, upper)
invertedmask1 = cv2.bitwise_not(mask1)
invertedmask2 = cv2.bitwise_not(mask2)

blurred_img = cv2.GaussianBlur(invertedmask1, (5, 5), 0)
blurred_img2 = cv2.GaussianBlur(invertedmask2, (5, 5), 0)

edge_detected_img1 = cv2.Canny(blurred_img, 50, 150)
edge_detected_img2 = cv2.Canny(blurred_img2, 50, 150)

contours, hierarchy = cv2.findContours(image=edge_detected_img1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(image=edge_detected_img2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

largest_contour_img1 = max(contours, key=cv2.contourArea)
largest_contour_img2 = max(contours2, key=cv2.contourArea)

rectangle_image1 = cv2.minAreaRect(largest_contour_img1)
rectangle_image2 = cv2.minAreaRect(largest_contour_img2)
box1 = cv2.boxPoints(rectangle_image1)
box2 = cv2.boxPoints(rectangle_image2)

box1 = np.int_(box1)
box2 = np.int_(box2)

image_copy = img1.copy()
image_copy2 = img2.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.drawContours(image=image_copy2, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.drawContours(image_copy, [box1], 0, (0, 255, 0), 2)
cv2.drawContours(image_copy2, [box2], 0, (0, 255, 0), 2)

x1, y1, w1, h1 = cv2.boundingRect(box1)
x2, y2, w2, h2 = cv2.boundingRect(box2)
image1_height = image_copy.shape[0]
image2_height = image_copy2.shape[0]

horizontal_midpt1 = int(x1 + (w1 / 2))
horizontal_midpt2 = int(x2 + (w2 / 2))

cv2.line(image_copy, (horizontal_midpt1, 0), (horizontal_midpt1, image1_height), (0, 0, 0), 2)
cv2.line(image_copy2, (horizontal_midpt2, 0), (horizontal_midpt2, image2_height), (0, 0, 0), 2)

int_coordinates_of_rectangle_image1 = []
for n in box2:
    int_coordinates_of_rectangle_image1.append(n)

coordinates1 = int_coordinates_of_rectangle_image1[0]
coordinates2 = int_coordinates_of_rectangle_image1[2]

y_end, y_start, usable_slope= line_equation_end_finder(coordinates1, coordinates2, image_copy2.shape[1])

cv2.line(image_copy, (250, 328), (242, 391), (255, 0, 255), 2)
cv2.line(image_copy2, (0, y_start), (image_copy2.shape[1], y_end), (255, 0, 255), 2)

angle__of_line_with_vertical =math.degrees(math.pi)- math.degrees(math.atan(-1/usable_slope))
print ("For Image 2 the angle of the dirrection of arrow adn verticalis:",angle__of_line_with_vertical)

cv2.imshow("Final Image 1", image_copy)            
cv2.imshow("Final Image 2", image_copy2)
cv2.waitKey(0)
cv2.destroyAllWindows()
