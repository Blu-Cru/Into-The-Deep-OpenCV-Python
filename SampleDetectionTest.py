import cv2
import numpy as np
import math

# 2.9 * 10-6 m per pixel
#
PIXELS_PER_INCH = 40
REF_CENTER_PIXELS = [800, 800]
REF_CENTER_INCHES = [-6.0, 13.0]
REAL_ROI_X = [-13, 13]
REAL_ROI_Y = [8, 24]
np_img_points = np.float32([[895, 607], [1155, 602],
    [870, 810], [1207, 805]])

def main():
    src = cv2.imread(r"images\chart\1.jpg")
    if src is None:
        print("Error: image not loaded")
        
    show("Src", src)

    undistorted = undistort(src)
    # show("undistorted", undistorted)

    wbCorrected = gray_world_white_balance(undistorted)

    show("Img corrected", wbCorrected)

    # homography
    transformed = doHomographyTransform(wbCorrected)

    hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)

    color = 0 # 0 for red, 1 for yellow, 2 for blue

    if color == 0:
        thresh1 = cv2.inRange(hsv, (0, 60, 20), (10, 255, 255))
        thresh2 = cv2.inRange(hsv, (150, 60, 20), (180, 255, 255))
        hsvThresh = cv2.bitwise_or(thresh1, thresh2)
    elif color == 1:
        hsvThresh = cv2.inRange(hsv, (10, 25, 60), (60, 255, 255))
    else:
        hsvThresh = cv2.inRange(hsv, (80, 80, 20), (150, 255,255))

    masked = cv2.bitwise_and(transformed, transformed, mask=hsvThresh)
    # show("masked", masked)

    blurred = cv2.GaussianBlur(transformed, (5, 5), 1)
    # show("blurred", blurred)

    # B, G, R = cv2.split(blurred)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # B = clahe.apply(B)
    # G = clahe.apply(G)
    # R = clahe.apply(R)

    # equalized = cv2.merge((B, G, R))
    # show("equalized", equalized)

    # edges = cv2.Canny(equalized, 80, 100)
    # show("equalized edges", edges)

    blurredEdges = cv2.Canny(transformed, 50, 100)
    show("Blurred edges", blurredEdges)
    
    # dilation
    dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(blurredEdges, dilationElement)
    show("dilated", dilated)
    
    # erosion
    # erosionElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # eroded = cv2.erode(dilated, erosionElement)
    # show("Eroded", eroded)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contourImage = np.zeros_like(src)
    cv2.drawContours(contourImage, contours, -1, (0, 255, 0), 2)

    rectImage = transformed.copy()

    validContours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000.0 or area > 7000.0:
            continue

        rect = cv2.minAreaRect(cnt)

        (centerx, centery) = rect[0]
        (width, height) = rect[1]
        angle = rect[2]

        # Check to avoid division by zero
        if width == 0 or height == 0:
            continue

        # Calculate the aspect ratio using the longer side divided by the shorter side
        ratio = max(width, height) / min(width, height)

        if ratio < 2.0 or ratio > 3.0:
            continue

        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) > 7:
            continue

        (centerx, centery) = getRealWorldCoords(centerx, centery)
        if centerx > REAL_ROI_X[1] or centerx < REAL_ROI_X[0] or centery > REAL_ROI_Y[1] or centery < REAL_ROI_Y[0]:
            continue
        
        if height > width:
            angle = 90-angle
        else:
            angle = -angle
        print(f'Contour with angle {angle}')
        print(f'Contour with width, height of {(width, height)}')
        print(f'Contour with center at {(centerx, centery)}')

        # Print the number of sides
        print(f'Contour with {len(approx)} sides')

        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(rectImage, [box], 0, (0,255, 0), 2)

        validContours.append(cnt)
            
    show("new contours", contourImage)
    show("Rects", rectImage)

    threshContours = []
    # for cnt in validContours:


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

def doHomographyTransform(src):
    imgPoints = []
    for point in np.ndarray.tolist(np_img_points):
        imgPoints.append((point[0], point[1]))
    print(imgPoints)

    points = src.copy()

    for (x, y) in imgPoints:
            print(x, y)
            cv2.circle(points, (int(x), int(y)), 5, (0, 0, 255), -1)
    show("Points", points)

    np_top_down_points = np.float32([[REF_CENTER_PIXELS[0], REF_CENTER_PIXELS[1]-PIXELS_PER_INCH * 5], [REF_CENTER_PIXELS[0]+PIXELS_PER_INCH*5, REF_CENTER_PIXELS[1]-PIXELS_PER_INCH*5],
        [REF_CENTER_PIXELS[0], REF_CENTER_PIXELS[1]], [REF_CENTER_PIXELS[0]+PIXELS_PER_INCH*5, REF_CENTER_PIXELS[1]]])
    
    M = cv2.getPerspectiveTransform(np_img_points,
                                    np_top_down_points)
    
    # each inch is 40 pixels
    top_down_size = (1920, 1080)  # (width, height)
    top_down_view = cv2.warpPerspective(src, M, top_down_size)

    # -------------------------------------------------------
    # 8) Save or display the result
    # -------------------------------------------------------
    show("Top-Down View", top_down_view)
    return top_down_view

def getExtrinsicRotation(yaw, pitch, roll):
    return np.matrix([
        [math.cos(pitch)*math.cos(roll), math.sin(yaw)*math.sin(pitch)*math.cos(roll)-math.cos(yaw)*math.sin(roll), math.cos(yaw)*math.sin(pitch)*math.cos(roll)+math.sin(yaw)*math.sin(roll)],
        [math.cos(pitch)*math.sin(roll), math.sin(yaw)*math.sin(pitch)*math.sin(roll)+math.cos(yaw)*math.cos(roll), math.cos(yaw)*math.sin(pitch)*math.sin(roll)+math.sin(yaw)*math.cos(roll)],
        [-math.sin(pitch), math.sin(yaw)*math.cos(pitch), math.cos(yaw)*math.cos(pitch)]
    ])

def compute_homography(K, R, camera_center):
    """
    Computes the 3x3 homography H that maps a point (X, Y, 1) on the z=0 plane
    in the world frame to the image plane, given K, R, and camera_center in world coords.
    """
    # camera_center is a 3D vector [Cx, Cy, Cz] in the world frame
    # translation (in the camera’s extrinsic) is t = -R @ C
    t = -R @ camera_center
    
    # Extract r1 and r2 (the first two columns of R),
    # then form a 3x3 by [r1, r2, t]
    r1 = R[:, 0]
    r2 = R[:, 1]
    # Stack them side by side into a 3x3
    R_2cols_t = np.column_stack((r1, r2, t))
    
    # Finally multiply by K to get the homography
    H = K @ R_2cols_t
    return H

def getRealWorldCoords(centerx, centery):
    refOffsetX = centerx-REF_CENTER_PIXELS[0]
    refOffsetY = -(centery-REF_CENTER_PIXELS[1])

    inchOffsetX = REF_CENTER_INCHES[0]+refOffsetX/PIXELS_PER_INCH
    inchOffsetY = REF_CENTER_INCHES[1]+refOffsetY/PIXELS_PER_INCH
    return (inchOffsetX, inchOffsetY)

main()