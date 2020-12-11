import cv2
import numpy as np
import mapper

image = cv2.imread("img.jpeg")  # Input image
image = cv2.resize(image, (1300, 800))  # resized
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converted to RGB

blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Blured the image

edged = cv2.Canny(blurred, 30, 50)  # Edge detection for the findContours function

contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

target = 0
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break
approx = mapper.mapp(target)

pts = np.float32([[0, 0], [1200, 0], [1200, 1200], [0, 1200]])

op = cv2.getPerspectiveTransform(approx, pts)
dst1 = cv2.warpPerspective(orig, op, (1200, 1200))
gray2 = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray2, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Scanned", thresh)
cv2.waitKey(0)
