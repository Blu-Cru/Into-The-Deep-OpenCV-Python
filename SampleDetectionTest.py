import cv2
import numpy as np

src = cv2.imread(r"images\1.jpg")
red = True
if src is None:
    print("Error: image not loaded")

cv2.imshow("Src", src)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

if red:
    thresh1 = cv2.inRange(hsv, (0, 40, 20), (10, 255, 255))
    thresh2 = cv2.inRange(hsv, (150, 40, 20), (180, 255, 255))
    hsvThresh = cv2.bitwise_or(thresh1, thresh2)
else:
    hsvThresh = cv2.inRange(hsv, (80, 40, 20), (140, 255,255))

masked = cv2.bitwise_and(src, src, mask=hsvThresh)

#dilation
dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dilated = cv2.dilate(src, dilationElement)
# cv2.imshow("dilated", dilated)

# erosion
element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

eroded = cv2.erode(src, element)
# cv2.imshow("Eroded", eroded)

edges = cv2.Canny(src, 100, 200)
cv2.imshow("edges", edges)

dilatedEroded = cv2.erode(dilated, element)
cv2.imshow("Both", dilatedEroded)

dilatedErodedEdges = cv2.Canny(dilatedEroded, 100, 200)
cv2.imshow("new edges", dilatedErodedEdges)

contours, hierarchy = cv2.findContours(dilatedErodedEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contourImage = np.zeros_like(src)
cv2.drawContours(contourImage, contours, -1, (0, 255, 0), 2)
cv2.imshow("new contours", contourImage)

cv2.waitKey(0)

cv2.destroyAllWindows()