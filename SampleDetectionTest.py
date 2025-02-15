import cv2

image = cv2.imread(r"C:\Users\kevin\Documents\GitHub\6417-Into-The-Deep-OpenCV\KELP1217.jpg")
red = True
if image is None:
    print("Error: image not loaded")

imSmall = cv2.resize(image, (600, 800))
hsv = cv2.cvtColor(imSmall, cv2.COLOR_BGR2HSV)

if red:
    thresh1 = cv2.inRange(hsv, (0, 40, 20), (20, 255, 255))
    thresh2 = cv2.inRange(hsv, (150, 40, 20), (180, 255, 255))
    hsvThresh = cv2.bitwise_or(thresh1, thresh2)
else:
    hsvThresh = cv2.inRange(hsv, (80, 40, 20), (140, 255,255))

cv2.imshow("hsv", hsvThresh)

cv2.waitKey(0)

cv2.destroyAllWindows()