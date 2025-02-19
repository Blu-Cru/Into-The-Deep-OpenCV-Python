import cv2
import numpy as np

src = cv2.imread(r"images\1.jpg")
if src is None:
    print("Error: image not loaded")

# cv2.imshow("Src", src)

rotated = cv2.rotate(src, cv2.ROTATE_180)

hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)

#dilation
dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dilated = cv2.dilate(rotated, dilationElement)
# cv2.imshow("dilated", dilated)

# erosion
erosionElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# eroded = cv2.erode(src, element)
# cv2.imshow("Eroded", eroded)

# edges = cv2.Canny(src, 100, 200)
# cv2.imshow("edges", edges)

dilatedEroded = cv2.erode(dilated, erosionElement)
cv2.imshow("Both", dilatedEroded)

color = 2 # 0 for red, 1 for yellow, 2 for blue

if color == 0:
    thresh1 = cv2.inRange(hsv, (0, 60, 20), (10, 255, 255))
    thresh2 = cv2.inRange(hsv, (150, 60, 20), (180, 255, 255))
    hsvThresh = cv2.bitwise_or(thresh1, thresh2)
elif color == 1:
    hsvThresh = cv2.inRange(hsv, (10, 100, 20), (70, 255, 255))
else:
    hsvThresh = cv2.inRange(hsv, (80, 110, 20), (140, 255,255))

masked = cv2.bitwise_and(dilatedEroded, dilatedEroded, mask=hsvThresh)
cv2.imshow("masked", masked)

edges = cv2.Canny(masked, 100, 200)
cv2.imshow("new edges", edges)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contourImage = np.zeros_like(src)
cv2.drawContours(contourImage, contours, -1, (0, 255, 0), 2)
cv2.imshow("new contours", contourImage)

cv2.waitKey(0)

cv2.destroyAllWindows()