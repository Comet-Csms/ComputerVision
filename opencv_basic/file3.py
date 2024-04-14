import cv2 as cv
import sys

img = cv.imread("./soccer.jpg")

if img is None:
    sys.exit("파일이 존재하지 않습니다.")

cv.rectangle(img, (550, 600), (750, 800), (0, 0, 255), 2) # 직사각형 그리기
cv.putText(img, "Ball", (550, 580), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 글씨 쓰기

cv.imshow("Draw", img)

cv.waitKey()
cv.destroyAllWindows()