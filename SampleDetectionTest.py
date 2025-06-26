import cv2
import numpy as np
import math

# 2.9 * 10-6 m per pixel
#
PIXELS_PER_INCH = 20
BOTTOM_LEFT_PIXELS = [800, 800]
BOTTOM_LEFT_INCHES = [-5.6, 13.0]
DIST_BETWEEN_POINTS = 5.6
np_img_points = np.float32([[853, 612], [1201, 612],
    [812, 947], [1287, 941]])
ALLIANCE = 2
RED_HUE_LOW_RANGE_1 = 20.0
RED_HUE_HIGH_RANGE_1 = 140.0
RED_HUE_LOW_RANGE_2 = 48
RED_HUE_HIGH_RANGE_2 = 100
YELLOW_HUE_LOW = 40.0
YELLOW_HUE_HIGH = 20.0
BLUE_HUE_LOW = 110.0
BLUE_HUE_HIGH = 120.0
MIN_X = 13


def main():
    src = cv2.imread(r"images\6.25\9.jpg")
    if src is None:
        print("Error: image not loaded")
        
    show("Src", src)

    undistorted = undistort(src)
    # show("undistorted", undistorted)

    # homography
    transformed = doHomographyTransform(undistorted)

    wbCorrected = gray_world_white_balance(transformed)

    

    show("Img corrected", wbCorrected)


    hsv = cv2.cvtColor(wbCorrected, cv2.COLOR_BGR2HSV)
    # show("Hue", h)
    # show("Blurred Hue", blurredH)

    channels = cv2.split(hsv)

    hueChannel = channels[0]
    satChannel = channels[1]

    saturationThresh = cv2.inRange(hsv, (0, 45, 0), (255, 255, 255))
    satMasked = cv2.bitwise_and(transformed, transformed, mask=saturationThresh)
    show("Saturation masked", satMasked)

    satEdges = cv2.Canny(satMasked,50,200)
    show("sat edges", satEdges)

    edges = cv2.Canny(wbCorrected,0,150)
    show("edges", edges)

    combinedEdges = cv2.bitwise_or(satEdges,edges)

    dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(combinedEdges, dilationElement)
    show("Dilated", dilated)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    detectionOverlay = wbCorrected.copy()

    validRects = []

    minDistance = float("inf")

    rectImage = transformed.copy()

    pose = (0,0,0)


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 750.0 or area > 1650.0:
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

        if ratio < 1.55 or ratio > 3:
           continue

        pts = cv2.boxPoints(rect)

        matOfRectPoints = np.array(pts)        
            
        if height > width:
            angle = 90-angle
        else:
            angle = -angle


        rotatedRectMask = np.zeros(wbCorrected.shape[:2], dtype=np.uint8)

        cv2.fillConvexPoly(rotatedRectMask, matOfRectPoints.astype(np.int32), 255)

        meanSat = cv2.mean(satChannel, rotatedRectMask)

        minSat = 70

        if (meanSat[0] < minSat):
            print(f'Contour with sat {meanSat[0]} discarded!\n')
            continue

        meanHue = cv2.mean(hueChannel, rotatedRectMask)



        print(f'Contour mean hue {meanHue[0]}')
        if ALLIANCE == 0 and (BLUE_HUE_LOW < meanHue[0] < BLUE_HUE_HIGH):
            print(f'Contour with hue {meanHue[0]} discarded!\n')
            continue
        elif ALLIANCE == 1 and (meanHue[0] < RED_HUE_LOW_RANGE_1 or meanHue[0] > RED_HUE_HIGH_RANGE_1 or (RED_HUE_LOW_RANGE_2 < meanHue[0] < RED_HUE_HIGH_RANGE_2)):
            print(f'Contour with hue {meanHue[0]} discarded!\n')
            continue



        # mask = np.zeros_like(transformed)
        # cv2.drawContours(mask,[cnt],0,255,-1)
        # pixelpoints = np.transpose(np.nonzero(mask))
        # print(pixelpoints)

        # print(f'Contour with angle {angle}')
        # print(f'Contour with width, height of {(width, height)}')

        #get robot point

        realPoint = getRealWorldCoords(centerx,centery)

        if (MIN_X > realPoint[1]):
            print(f'Contour with x {realPoint[1]} discarded!\n')
            continue

# print sat        
        #cv2.putText(rectImage, f"Sat: {int(meanSat[0])}", tuple(map(int, rect[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        # print hue
        #cv2.putText(rectImage, f"Hue: {int(meanHue[0])}", tuple(map(int, rect[0])), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 2)
        # # print area
        #cv2.putText(rectImage, f"Area: {int(area)}", tuple(map(int, rect[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        # # print ratio
        #cv2.putText(rectImage, f"Ratio: {round(ratio,2)}", tuple(map(int, rect[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        #     # // print center
        #cv2.putText(rectImage, f"({int(realPoint[0])}, {int(realPoint[1])}), {int(angle)}", tuple(map(int, rect[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)


        # Print the number of sides
        # print(f'Contour with {len(approx)} sides')

        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(rectImage, [box], 0, (0,255, 0), 2)

        validRects.append(cnt)
            
    show("new contours", rectImage)
    # show("Rects", rectImage)
    # show("Detection overlay", detectionOverlay)

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

    np_top_down_points = np.float32([[BOTTOM_LEFT_PIXELS[0], BOTTOM_LEFT_PIXELS[1]-PIXELS_PER_INCH * DIST_BETWEEN_POINTS], [BOTTOM_LEFT_PIXELS[0]+PIXELS_PER_INCH*DIST_BETWEEN_POINTS, BOTTOM_LEFT_PIXELS[1]-PIXELS_PER_INCH*DIST_BETWEEN_POINTS],
        [BOTTOM_LEFT_PIXELS[0], BOTTOM_LEFT_PIXELS[1]], [BOTTOM_LEFT_PIXELS[0]+PIXELS_PER_INCH*DIST_BETWEEN_POINTS, BOTTOM_LEFT_PIXELS[1]]])
    
    M = cv2.getPerspectiveTransform(np_img_points,
                                    np_top_down_points)
    
    # each inch is 20 pixels
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
    refOffsetX = centerx-BOTTOM_LEFT_PIXELS[0]
    refOffsetY = -(centery-BOTTOM_LEFT_PIXELS[1])

    inchOffsetX = BOTTOM_LEFT_INCHES[0]+refOffsetX/PIXELS_PER_INCH
    inchOffsetY = BOTTOM_LEFT_INCHES[1]+refOffsetY/PIXELS_PER_INCH
    return (inchOffsetX, inchOffsetY)

main()