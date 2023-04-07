# 霍夫变化及svm向量机包
import joblib
import skimage.feature
import cv2
from skimage.feature import hog

clf = joblib.load("model_svm/train_model.m")
# roi = cv2.imread('hogsvm/data/test/yes/0552.jpg',0)
roi = cv2.imread('./112/0645.jpg',0)
# print(roi.shape)
# path_name = r'112'
# for item in os.listdir(path=path_name):
if len(roi.shape) == 3:
    gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (128, 128))
else:
    gray = cv2.resize(roi, (128, 128))
# roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# svm_roi = cv2.resize(roi, (128, 128))
# 因为每个cellsize有8个特征，所以每个块内有1*8 = 8个特征，以16个像素为步长（pixels_per_cell=（16，16）），那么，水平方向将有16个扫描窗口，垂直方向将有16个扫描窗口。
# 也就是说，64x128的图片，总共有8 x 16 x 16 = 2048个特征。
# v,hf = skimage.feature.hog(svm_roi, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualize=True)
hf = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
# 预测结果 0为存在电表1为不存在
# print(hf.shape)
# clf = joblib.load("train_model.m")
# hf = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
svm_result = clf.predict([hf])
print(f"识别结果为{'no' if svm_result[0] else 'yes'}")
# svm_result = clf.predict(hf)
print(svm_result)
# cv2.imshow('hog',hf)
# cv2.waitKey(0)
