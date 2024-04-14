import skimage
import numpy as np
import cv2 as cv
import time

img = skimage.data.coffee()

start = time.time()

slic = skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)

g = skimage.graph.rag_mean_color(img, slic, mode="similarity")
ncut = skimage.graph.cut_normalized(slic, g)
print(img.shape, "image를 분할하는 데", time.time()-start, "초 소요")

marking = skimage.segmentation.mark_boundaries(img, ncut)
ncut_coffee = np. uint8 (marking*255.0)

cv.imshow("Normalized cut", cv.cvtColor(ncut_coffee, cv.COLOR_BGR2RGB))

cv.waitKey()
cv.destroyAllWindows()