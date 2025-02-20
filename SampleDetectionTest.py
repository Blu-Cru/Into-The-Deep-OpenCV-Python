import cv2
import numpy as np

def main():
    src = cv2.imread(r"images\1.jpg")
    if src is None:
        print("Error: image not loaded")
        
    show("Src", src)

    undistorted = undistort(src)
    show("undistorted", undistorted)

    wbCorrected = gray_world_white_balance(undistorted)

    show("Img corrected", wbCorrected)

    hsv = cv2.cvtColor(wbCorrected, cv2.COLOR_BGR2HSV)

    # erosion
    # erosionElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # dilatedEroded = cv2.erode(dilated, erosionElement)
    # show("Both", dilated)

    color = 2 # 0 for red, 1 for yellow, 2 for blue

    if color == 0:
        thresh1 = cv2.inRange(hsv, (0, 60, 20), (15, 255, 255))
        thresh2 = cv2.inRange(hsv, (150, 60, 20), (180, 255, 255))
        hsvThresh = cv2.bitwise_or(thresh1, thresh2)
    elif color == 1:
        hsvThresh = cv2.inRange(hsv, (15, 100, 20), (70, 255, 255))
    else:
        hsvThresh = cv2.inRange(hsv, (80, 100, 20), (150, 255,255))

    masked = cv2.bitwise_and(wbCorrected, wbCorrected, mask=hsvThresh)
    show("masked", masked)

    blurred = cv2.GaussianBlur(masked, (5, 5), 1)
    show("blurred", blurred)

    # B, G, R = cv2.split(blurred)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # B = clahe.apply(B)
    # G = clahe.apply(G)
    # R = clahe.apply(R)

    # equalized = cv2.merge((B, G, R))
    # show("equalized", equalized)

    # edges = cv2.Canny(equalized, 80, 100)
    # show("equalized edges", edges)

    blurredEdges = cv2.Canny(blurred, 20, 100)
    show("Blurred edges", blurredEdges)
    
    # dilation
    dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(blurredEdges, dilationElement)
    show("dilated", dilated)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourImage = np.zeros_like(src)
    cv2.drawContours(contourImage, contours, -1, (0, 255, 0), 2)
    show("new contours", contourImage)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def show(str, mat):
    cv2.imshow(str, cv2.resize(mat, (960, 540)))

def undistort(img):
    h, w = img.shape[:2]

    cameraMatrix = np.array([[1279.33,   0, 958.363], 
                             [  0, 1279.33, 492.062], 
                             [  0,   0,   1]], dtype=np.float64)
    
    distCoeffs = np.array([-0.448017, 0.245668, -0.000901464, 0.000996399], dtype=np.float64)

    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, 
        distCoeffs, 
        (w, h), 
        alpha=1, 
        newImgSize=(w, h)
    )

    return cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)

def gray_world_white_balance(img):
    # Calculate the mean of each channel
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    # Calculate scaling factors for each channel
    avg = (avg_b + avg_g + avg_r) / 3
    scale_b = avg / avg_b
    scale_g = avg / avg_g
    scale_r = avg / avg_r

    # Apply scaling factors to the image
    img[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)

    return img

main()