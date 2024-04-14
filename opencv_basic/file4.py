import cv2 as cv
import sys

BrushSize = 5
LColor, RColor = (255, 0, 0), (0, 0, 255)

img = cv.imread("./apples.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), BrushSize, LColor, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), BrushSize, RColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), BrushSize, LColor, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), BrushSize, RColor, -1)

    cv.imshow("Painting", img)

cv.namedWindow("Painting")
#cv.imshow("Painting", img)
cv.setMouseCallback("Painting", painting)

while(True): # 마우스 이벤트가 언제 발생할지 모르므로 무한 반복
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break