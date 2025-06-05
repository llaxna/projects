import cv2
import numpy as np

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def get_biggest_contour(contours, min_area=5000):
    biggest = np.array([])
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

webCam = True
cap = cv2.VideoCapture(1)
cap.set(10, 0)  # brightness
height = 640
width = 480

imgWarp = None  

while True:
    imgBlank = np.zeros((height, width, 3), np.uint8)
    imgWarp = None
    if webCam:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (width, height))
    else:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThresh = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    imgCanny = cv2.Canny(imgThresh, 50, 150)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDil, kernel, iterations=1)

    contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Take largest 10

    biggest = get_biggest_contour(contours)

    if biggest.size != 0:
        cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (width, height))
        cv2.imshow("Scanned Document", imgWarp)
    else:
        cv2.imshow("Scanned Document", imgBlank)

    # Debug windows
    cv2.imshow("Original", img)
    cv2.imshow("Gray", imgGray)
    cv2.imshow("Threshold", imgThresh)
    cv2.imshow("Canny", imgCanny)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s') and imgWarp is not None:
        cv2.imwrite("ScannedDocument.jpg", imgWarp)
        print("Scanned document saved as ScannedDocument.jpg")

cap.release()
cv2.destroyAllWindows()
