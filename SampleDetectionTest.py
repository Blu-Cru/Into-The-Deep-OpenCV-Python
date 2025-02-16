import cv2

src = cv2.imread(r"images\1.jpg")
red = True
if src is None:
    print("Error: image not loaded")

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

if red:
    thresh1 = cv2.inRange(hsv, (0, 40, 20), (20, 255, 255))
    thresh2 = cv2.inRange(hsv, (150, 40, 20), (180, 255, 255))
    hsvThresh = cv2.bitwise_or(thresh1, thresh2)
else:
    hsvThresh = cv2.inRange(hsv, (80, 40, 20), (140, 255,255))

masked = cv2.bitwise_and(src, src, hsvThresh)

cv2.imshow("Out", hsvThresh)

cv2.waitKey(0)

cv2.destroyAllWindows()