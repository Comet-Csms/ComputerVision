import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# 모델 선택
model = ResNet50(weights="imagenet") # ImageNet으로 학습한 ResNet50을 백본으로 사용

# 이미지 전처리
img = cv.imread("./rabbit.jpg")
x = np.reshape(cv.resize(img, (224,224)), (1,224,224,3))
x = preprocess_input(x)

# 이미지 예측
preds = model.predict(x)
top5 = decode_predictions(preds, top=5)[0]
print("예측 결과:", top5)

# 이미지 출력 with 예측 결과
for i in range(5):
    cv.putText(img, top5[i][1]+':'+str(top5[i][2]), (10, 20+i*20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

cv.imshow("Recognition result", img)

cv.waitKey()
cv.destroyAllWindows()