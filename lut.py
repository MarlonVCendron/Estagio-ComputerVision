import cv2
import numpy as np

invGamma = 0.2
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
txt = "["
for i in table:
	txt += str(i) + ", "
print(txt)

# img = cv2.imread('flower.jpg')
# brighter_img = cv2.LUT(img, table)
# while True:
# 	cv2.imshow('normal', img)
# 	cv2.imshow('gama alterado', brighter_img)
# 	if cv2.waitKey(30) & 0xff == ord('a'):
# 		break
#
# cv2.destroyAllWindows()