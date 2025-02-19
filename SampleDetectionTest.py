import cv2
import numpy as np

def main():
    src = cv2.imread(r"images\1.jpg")
    if src is None:
        print("Error: image not loaded")

    show("Src", src)

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    #dilation
    # dilationElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # dilated = cv2.dilate(src, dilationElement)
    # cv2.imshow("dilated", dilated)

    # erosion
    # erosionElement = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # dilatedEroded = cv2.erode(dilated, erosionElement)
    # show("Both", dilated)

    color = 1 # 0 for red, 1 for yellow, 2 for blue

    if color == 0:
        thresh1 = cv2.inRange(hsv, (0, 60, 20), (15, 255, 255))
        thresh2 = cv2.inRange(hsv, (150, 60, 20), (180, 255, 255))
        hsvThresh = cv2.bitwise_or(thresh1, thresh2)
    elif color == 1:
        hsvThresh = cv2.inRange(hsv, (15, 100, 20), (70, 255, 255))
    else:
        hsvThresh = cv2.inRange(hsv, (80, 40, 20), (150, 255,255))


    masked = cv2.bitwise_and(src, src, mask=hsvThresh)
    show("masked", masked)

    blurred = cv2.GaussianBlur(masked, (5, 5), 1)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)

    edges = cv2.Canny(equalized, 50, 100)
    show("new edges", edges)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourImage = np.zeros_like(src)
    cv2.drawContours(contourImage, contours, -1, (0, 255, 0), 2)
    show("new contours", contourImage)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def show(str, mat):
    cv2.imshow(str, cv2.resize(mat, (960, 540)))

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

main()