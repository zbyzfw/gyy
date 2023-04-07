# import os
# os.environ['SETUPTOOLS_USE_DISTUTILS'] = "stdlib"
# import cv2
# import sys, json
# import time, re
# import glob
# import warnings
# import configparser
# from xml.dom.minidom import Document
# from lxml import etree, objectify
# from lxml.etree import Element
# import traceback
# import logging
# import datetime
# from tools.infer.predict import detect_roi
#
#
# warnings.filterwarnings("ignore")
# if not os.path.exists('barLog'):
#     os.makedirs('barLog')
# logfilename = f"barLog/Log{datetime.date.today().strftime('%Y%m%d')}.log"
# logging.basicConfig(level=logging.INFO, filename=logfilename, format="%(asctime)s: %(module)s: %(message)s")
# cf = configparser.ConfigParser()
#
#
# # img_path = sys.argv[1]
# # ini_path = sys.argv[2]
# # rotate = sys.argv[3]
# # type_name = sys.argv[4]
# # number_e= sys.argv[5]
# img_path = './photo/img1'
# ini_path = './ini'
# rotate = '1'
# type_name = '3'
# number_e= '6'
# #20版三相
# fp = open('meter.json', encoding='utf-8')
# meter_data = json.load(fp)
# meter1 = meter_data["20版三相"]
# meter2 = meter_data["20版单相"]
# meter3 = meter_data["13版三相"]
# meter4 = meter_data["13版单相"]
# # 实例化一个xml节点
# E = objectify.ElementMaker(annotate=False)
#
# def detect_lcd(img_path,ini_path, rotate, type_name, e,number_e):
#     # result_list = []
#     # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
#     img = cv2.imread(img_path)
#
#     # print('图片信息:', img.shape)
#     # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         gray = img.copy()
#     # for i in range(1, len(cf.sections()) + 1):
#     cf.get=ini_path
#     top_x = int(cf.get(f"Location1", "topX"))
#     top_y = int(cf.get(f"Location1", "topY"))
#     width = int(cf.get(f"Location1", "width"))
#     height = int(cf.get(f"Location1", "height"))
#     try:
#         roi = img[top_y:top_y + height, top_x:top_x + width, ]
#     except Exception as e:
#         tree41 = E.BAR_CODE(f"{path} ini error")
#         tree42 = E.CHECK_RESULT(
#             E.REGION_TYPE("1"),
#             E.CHECK_FLAG("NG")
#         )
#         child1.append(tree41)
#         child1.append(tree42)
#         return
#     if rotate == '1':
#         roi = cv2.flip(cv2.transpose(roi), 1)
#     elif rotate == '2':
#         roi = cv2.flip(roi, -1)
#     elif rotate == '3':
#         roi = cv2.flip(cv2.transpose(cv2.flip(roi, -1)), 1)
#     set_meter = []
#     results = detect_roi(roi)
#     # print(results)
#     if results:
#         # 识别结果转换为列表
#         results_news = [res[0] for res in results]
#         # for res in results:
#         if type_name == '1':
#             # 替换识别错误的字符
#             # print(meter1[-1].split(','))
#             for meter in meter1[-1].split(','):
#                 if meter:
#                     temp = meter.split('>')
#                     results_news = [re.sub(temp[0],temp[1],res) for res in results_news]
#             # print(results_news)
#             for res in meter1[:-1]:
#                 if res not in results_news:
#                     # print(res)
#                     set_meter.append(res)
#         elif type_name == '2':
#             for meter in meter2[-1].split(','):
#                 if meter:
#                     temp = meter.split('>')
#                     results_news = [res.replace(temp[0], temp[1]) for res in results_news]
#
#             for res in meter2[:-1]:
#                 if res not in results_news:
#                     set_meter.append(res)
#         elif type_name == '3':
#             for meter in meter3[-1].split(','):
#                 if meter:
#                     temp = meter.split('>')
#                     results_news = [res.replace(temp[0], temp[1]) for res in results_news]
#
#             for res in meter3[:-1]:
#                 if res not in results_news:
#                     set_meter.append(res)
#         elif type_name == '4':
#             for meter in meter4[-1].split(','):
#                 if meter:
#                     temp = meter.split('>')
#                     results_news = [res.replace(temp[0], temp[1]) for res in results_news]
#
#             for res in meter4[:-1]:
#                 if res not in results_news:
#                     set_meter.append(res)
#         else:
#             E = objectify.ElementMaker(annotate=False)
#             E = objectify.ElementMaker(annotate=False)
#             result_xml = E.DATA(
#                 E.METER(
#                     E.BAR_CODE(bars),
#                     E.CHECK_RESULT(
#                         E.REGION_TYPE("1"),
#                         E.CHECK_FLAG(f"NG,请输入要识别的电表类型")
#                     ),
#                 ),
#             )
#             etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
#
#         # print(set_meter)
#         if set_meter:
#             E = objectify.ElementMaker(annotate=False)
#             result_xml = E.DATA(
#                 E.METER(
#                     E.BAR_CODE(bars),
#                     E.CHECK_RESULT(
#                         E.REGION_TYPE("1"),
#                         E.CHECK_FLAG(f"NG,缺字")
#                     ),
#                 ),
#             )
#             etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
#         else:
#             E = objectify.ElementMaker(annotate=False)
#             result_xml = E.DATA(
#             E.METER(
#                 E.BAR_CODE(bars),
#                 E.CHECK_RESULT(
#                     E.REGION_TYPE("1"),
#                     E.CHECK_FLAG(f"OK")
#                 ),
#             ),
#         )
#         result_xml.append(result_xml1)
#         etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
#         # cv2.imshow("3", roi)
#         # cv2.waitKey(1000)
#     # print(set_meter)
#
#     else:
#         E = objectify.ElementMaker(annotate=False)
#         result_xml = E.DATA(
#             E.METER(
#                 E.BAR_CODE(bars),
#                 E.CHECK_RESULT(
#                     E.REGION_TYPE("1"),
#                     E.CHECK_FLAG(f"NG,BLACK")
#                 ),
#             ),
#         )
#         etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
#     return E, results_xml
#
# def run(img_path, ini_path, rotate, type_name, number_e):
#     try:
#         E, result_xml = None, None
#         # 循环读取模板
#         for i in range(int(number_e)):
#             img_path_temp = img_path + f'/PICTURE{i+1}\*.jpg'
#             imgs = glob.glob(img_path_temp)  # 所有目录
#             if range_d.find('1') != -1:
#                 ini_path = ini_path + f'/config{i + 1}.ini'
#             else:
#                ini_path=''
#                img_path=''
#             E,result_xml=detect_lcd(imgs,ini_path,rotate, type_name, number_e,e,result_xml)
    # except Exception as e:
    #     E = objectify.ElementMaker(annotate=False)
    #     result_xml = E.DATA(
    #         E.METER(
    #             E.CHECK_RESULT(
    #                 E.REGION_TYPE("template search error"),
    #             ),
    #         ),
    #     )
    #     etree.ElementTree(result_xml).write("result.xml", pretty_print=True)


# if __name__ == '__main__':
    #
    # start = time.time()
    # root = Element("DATA")
    # child1 = Element("METER")
    # if not os.path.exists(img_path):
    #     # print('文件夹不存在')
    #     tree41 = E.BAR_CODE(f"{img_path}")
    #     tree42 = E.CHECK_RESULT(
    #         E.REGION_TYPE("1"),
    #         E.CHECK_FLAG(f"NG, {img_path}文件夹不存在，请输入包含图片的文件夹")
    #     )
    #     child1.append(tree41)
    #     child1.append(tree42)
    #
    # elif not os.path.exists(ini_path):
    #     # print('配置文件不存在')
    #     tree41 = E.BAR_CODE(f"{img_path}")
    #     tree42 = E.CHECK_RESULT(
    #         E.REGION_TYPE("1"),
    #         E.CHECK_FLAG(f"NG, {img_path}配置文件不存在，请输入正确配置文件路径")
    #     )
    #     child1.append(tree41)
    #     child1.append(tree42)
    # else:
    #     imgs = glob.glob(img_path+'/*.jpg')  # 所有目录
    #     # 创建xml文件根节点
    #     cf.read(ini_path)
    #     for img in imgs:
    #         detect_lcd(img, child1)
    # root.append(child1)
    # result = etree.ElementTree(root)
    # result.write('result.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')
    # # except:
    #     s = traceback.format_exc()
    #     logging.error(s)
    #     doc = Document()
    #     nodeError = doc.createElement('ERROR')
    #     doc.appendChild(nodeError)
    #     nodeError.appendChild(doc.createTextNode(f"An unknown error has occurred, see details in {logfilename}"))
    #     fp = open('result.xml', 'w')
    #     doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='UTF-8')
    #     fp.close()
    # print('总耗时',time.time()-start)
# if __name__ == '__main__':
#     run(img_path, ini_path, rotate, type_name, number_e)
# except Exception as e:
#     tree41 = E.BAR_CODE(f"{img_path}")
#     tree42 = E.CHECK_RESULT(
#         E.REGION_TYPE("1"),
#         E.CHECK_FLAG(f"NG, 未知错误,请联系管理员{e}")
#     )
#     child1.append(tree41)
#     child1.append(tree42)
#     root.append(child1)
#     result = etree.ElementTree(root)
#     result.write('result.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')
# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard imports
import os

# import cv2
# import numpy as np
# #
# # # path = r"D:\ocr\paddleocr\photo\img6\PICTURE4"
# #
# # # files = os.listdir(path)
# # # for file in files:
# #
# #     # Read image
# im = cv2.imread('./photo/img6/PICTURE4/0a.jpg')
# cv2.imshow("Keypoints", im)
# cv2.waitKey(1000)
# # im_o = cv2.resize(im, (600, 600))
#
# im_gauss = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# im_gauss = cv2.GaussianBlur(im_gauss, (3, 3), 0)
# ret, im = cv2.threshold(im_gauss, 170, 255, 0)
# # ret, im = cv2.threshold(im_gauss, 0, 111, 0)
# # cv2.imshow("o", im)
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 5
# params.maxThreshold = 10
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 5

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.3
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.57
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3:
#     detector = cv2.SimpleBlobDetector(params)
# else:
#     detector = cv2.SimpleBlobDetector_create(params)
#
# # Detect blobs.
# keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255, 255, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     # Show blobs
# im_o = cv2.resize(im_with_keypoints, (600, 600))
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(10000)
# cv2.imwrite('01.jpg',im_with_keypoints)
# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np
# def statistics():
#     src = cv.imread("photo/img6/PICTURE4/R1707.jpg")
#     cv.imshow("q",src)
#     h,w,ch = np.shape(src)
#     #读取图像属性
#     gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#     #将图像转换成灰度图，
#     cv.imshow("gray",gray)
#     hest = np.zeros([256],dtype = np.int32)
#     #建立空白数组
#     for row in range(h):
#         for col in range(w):
#             pv = gray[row,col]
#             hest[pv] +=1
#             #统计不同像素值出现的频率
#     plt.plot(hest,color = "r")
#     plt.xlim([0,256])
#     plt.show()
#     #画出统计图
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# statistics()
# !/usr/bin/python
# import random
#
# nums = range(1, 254)
# random.shuffle(nums)
#
# for i in range(0, 128, 2):
#     print(random.randrange(20, 220), nums[i], nums[i + 1])
# import random
# import os
# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont
#
#
# def createcode():
#     """
#     生成随机码
#     :return: 5位数随机码
#     """
#     s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     code = ''
#
#     for i in range(5):
#         c = s[int(random.random() * len(s))]
#         code += c
#
#     return code
#
#
# def createcodeimg(code, path):
#     """
#     生成随机码图片
#     :param code: 随机码
#     :param path: 保存路径
#     :return:
#     """
#     xoffset = 0
#     yoffset = 0
#     space = 20
#
#     img = Image.new('RGB', (0, 0), 'gray')
#     font = ImageFont.truetype("hyzsf.TTF", 100)
#
#     # 计算随机码长度
#     size = font.getsize(code)
#     w = size[0] + space * (len(code) + 1)
#     newsize = (w, size[1] + space * 2)
#
#     # 重置图片大小
#     img = img.resize(newsize)
#
#     # 计算纵向偏移，使纵向居中（不完全居中）
#     yoffset = (newsize[1] - size[1]) / 2
#
#     imgdraw = ImageDraw.Draw(img)
#
#
#     # 随机生成颜色
#     def randomcolor():
#         """
#         随机生成颜色
#         :return: 32位颜色
#         """
#         color = 0
#         for i in range(3):
#             color = (color << 8) | (int(random.random() * 0xff))
#         color = (0xff << 24) | color
#         # print '0x%x' % color
#         return color
#
#
#     # 随机生成位置
#     def randomxy():
#         """
#         基于图片的尺寸随机生成位置
#         :return: 位置tuple
#         """
#         x = int(random.random() * newsize[0])
#         y = int(random.random() * newsize[1])
#         return (x, y)
#
#
#     # draw lines
#     for i in range(10):
#         color = randomcolor()
#         imgdraw.line([randomxy(), randomxy()], fill=color, width=2)
#
#
#     # draw dots
#     for i in range(50):
#         xy = randomxy()
#         radius = 2
#         xy2 = (xy[0] + radius * 2, xy[1] + radius * 2)
#         imgdraw.ellipse([xy, xy2], randomcolor())
#
#
#     # draw code
#     lastdirection = None
#     for c in code:
#         size = font.getsize(c)
#         w, h = size[0], size[1]
#         r = max(w, h)
#         imgtmp = Image.new('RGBA', (r, r), 0)
#         imgdrawtmp = ImageDraw.Draw(imgtmp)
#         imgdrawtmp.text(((r - w) / 2, (r - h) / 2), c, fill=randomcolor(), font=font)
#         direction = 1 if random.random() > 0.5 else -1
#         #调整旋转方向
#         if lastdirection == direction:
#             lastdirection = direction = -direction
#         rotate = direction * (10 + random.random() * 45)
#         # 旋转
#         imgtmp = imgtmp.rotate(rotate)
#         imgdraw.bitmap((xoffset, yoffset), imgtmp, fill=randomcolor())
#
#         xoffset += (w + space)
#
#     # 保存到
#     img.save(path)
#
# if not os.path.exists('photo/img6/PICTURE4'):
#     os.makedirs('photo')
#
# for i in range(100):
#     # code = u'hello中国'
#     code = createcode()
#     createcodeimg(code, 'photo/img6/PICTURE4/%d.jpg' % i)
# import cv2
# import numpy as np
# img0= cv2.imread("photo/img6/PICTURE4/R1707.jpg")
#
# sss=np.zeros([480,640],dtype=np.uint8)
# sss[300:350,310:400]=255
# image=cv2.add(img0, np.zeros(np.shape(img0), dtype=np.uint8), mask=sss)
# cv.waitKey(0)
# cv.destroyAllWindows()
# import random
#
# import itertools
#
# import os
# import PIL.Image as Image
# import PIL.ImageDraw as ImageDraw
#
# # 原始图片的存放位置
# PATH = './photo/img6/PICTURE4/'
# # 新生成的图片的保存位置
# SAVE_PATH= './photo/'
# # 要在图片上生成几边形的物体，N=5代表五边形
# N = 3
#
#
# def drawObs(path, savePath, n):
#
#     for file in os.listdir(path):
#         if not file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
#             continue
#         img = Image.open(path + file)
#         (x, y) = img.size
#         random_list = list(itertools.product(range(1, x), range(1, y)))
#         points = random.sample(random_list, n)
#
#         x1 = random.randint(0, 255)
#         x2 = random.randint(0, 255)
#         x3 = random.randint(0, 255)
#
#         draw = ImageDraw.Draw(img)
#
#         ImageDraw.ImageDraw.polygon(xy=points, fill=(x1, x2, x3), self=draw)
#         img.save(savePath + 'rand' + file)
#         print(file)
#
# if __name__ == '__main__':
#     drawObs(PATH, SAVE_PATH, N)

# import cv2
# import math
# import random
#
# import matplotlib.image as im
# import numpy as np
#
#
# # def randomErasing(imgDir, sl=0.002, sh=0.004, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
# def randomErasing(imgDir,  r1=0.3):
#     img = cv2.imread(imgDir)
#     area = img.shape[0] * img.shape[1]
#     sl = random.uniform(0.001, 0.002)
#     sh = random.uniform(0.0001, 0.0009)
#
#
#     target_area = random.uniform(sl, sh) * area
#     aspect_ratio = random.uniform(r1, 1 / r1)
#     mean1 = np.random.randint(255, size=3).tolist()
#     mean2 = np.random.randint(255, size=3).tolist()
#     # mean3 = np.random.randint(255, size=3).tolist()
#     # mean4 = np.random.randint(255, size=3).tolist()
#     # mean5 = np.random.randint(255, size=3).tolist()
#     # mean6 = np.random.randint(255, size=3).tolist()
#     # mean7 = np.random.randint(255, size=3).tolist()
#     # mean8 = np.random.randint(255, size=3).tolist()
#     # mean9 = np.random.randint(255, size=3).tolist()
#     # mean1 =[255,255,255]
#     # mean2 =[255, 255, 255]
#     # mean3 = [255, 255, 255]
#     # mean4 = [255, 255, 255]
#     # mean5 = [255, 255, 255]
#     # mean6 = [255, 255, 255]
#     # mean7 = [255, 255, 255]
#     # mean8 = [255, 255, 255]
#     # mean9 = [255, 255, 255]
#
#
#     h = int(round(math.sqrt(target_area * aspect_ratio)))
#     w = int(round(math.sqrt(target_area / aspect_ratio)))
#     # print(mean9)
#
#     if w < img.shape[1] and h < img.shape[0]:
#         x1 = random.randint(0, img.shape[0] - h)
#         y1 = random.randint(0, img.shape[1] - w)
#         x2 = random.randint(0, img.shape[0] - h)
#         y2 = random.randint(0, img.shape[1] - w)
#         # x3 = random.randint(0, img.shape[0] - h)
#         # y3 = random.randint(0, img.shape[1] - w)
#         # x4 = random.randint(0, img.shape[0] - h)
#         # y4 = random.randint(0, img.shape[1] - w)
#         # x5 = random.randint(0, img.shape[0] - h)
#         # y5 = random.randint(0, img.shape[1] - w)
#         # x6 = random.randint(0, img.shape[0] - h)
#         # y6 = random.randint(0, img.shape[1] - w)
#         # x7 = random.randint(0, img.shape[0] - h)
#         # y7 = random.randint(0, img.shape[1] - w)
#         # x8 = random.randint(0, img.shape[0] - h)
#         # y8 = random.randint(0, img.shape[1] - w)
#         # x9 = random.randint(0, img.shape[0] - h)
#         # y9 = random.randint(0, img.shape[1] - w)
#         if img.shape[2] == 3:
#             img[x1:x1 + h, y1:y1 + w, 0] = mean1[0]
#             img[x1:x1 + h, y1:y1 + w, 1] = mean1[1]
#             img[x1:x1 + h, y1:y1 + w, 2] = mean1[2]
#             img[x2:x2 + h, y2:y2 + w, 0] = mean2[0]
#             img[x2:x2 + h, y2:y2 + w, 1] = mean2[1]
#             img[x2:x2 + h, y2:y2 + w, 2] = mean2[2]
#             # img[x3:x3 + h, y3:y3 + w, 0] = mean3[0]
#             # img[x3:x3 + h, y3:y3 + w, 1] = mean3[1]
#             # img[x3:x3 + h, y3:y3 + w, 2] = mean3[2]
#             # img[x4:x4 + h, y4:y4 + w, 0] = mean4[0]
#             # img[x4:x4 + h, y4:y4 + w, 1] = mean4[1]
#             # img[x4:x4 + h, y4:y4 + w, 2] = mean4[2]
#             # img[x5:x5 + h, y5:y5 + w, 0] = mean5[0]
#             # img[x5:x5 + h, y5:y5 + w, 1] = mean5[1]
#             # img[x5:x5 + h, y5:y5 + w, 2] = mean5[2]
#             # img[x6:x6 + h, y6:y6 + w, 0] = mean5[0]
#             # img[x6:x6 + h, y6:y6 + w, 1] = mean6[1]
#             # img[x6:x6 + h, y6:y6 + w, 2] = mean6[2]
#             # img[x7:x7 + h, y4:y4 + w, 0] = mean7[0]
#             # img[x7:x7 + h, y4:y4 + w, 1] = mean7[1]
#             # img[x7:x7 + h, y4:y4 + w, 2] = mean7[2]
#             # img[x8:x8 + h, y5:y5 + w, 0] = mean8[0]
#             # img[x8:x8 + h, y5:y5 + w, 1] = mean8[1]
#             # img[x8:x8 + h, y5:y5 + w, 2] = mean8[2]
#             # img[x9:x9 + h, y6:y6 + w, 0] = mean9[0]
#             # img[x9:x9 + h, y6:y6 + w, 1] = mean9[1]
#             # img[x9:x9 + h, y6:y6 + w, 2] = mean9[2]
#
#         else:
#             img[x1:x1 + h, y1:y1 + w, 0] = mean1[0]
#             img[x2:x2 + h, y2:y2 + w, 0] = mean2[0]
#             # img[x3:x3 + h, y3:y3 + w, 0] = mean3[0]
#             # img[x4:x4 + h, y4:y4 + w, 0] = mean4[0]
#             # img[x5:x5 + h, y5:y5 + w, 0] = mean5[0]
#             # img[x6:x6 + h, y6:y6 + w, 2] = mean6[2]
#             # img[x7:x7 + h, y7:y7 + w, 0] = mean7[0]
#             # img[x8:x8 + h, y8:y8 + w, 0] = mean8[0]
#             # img[x9:x9 + h, y9:y9 + w, 2] = mean9[2]
#
#     return img
#
#
# if __name__ == '__main__':
#     # imgDir = './0001.jpg'
#     # img = cv2.imread(imgDir)
#     # path_name = r'D:\ocr\paddleocr\train'
#     # for item in os.listdir(path=path_name):
#     #     # img = im.imread(os.path.join(path_name, item))
#     #     img = randomErasing(item)
#     #     cv2.imshow('show', img)
#     #     cv2.waitKey(0)
#     # cv2.imshow('show', img)
#     # cv2.waitKey(0)
#     # imagelist = os.listdir('D:/ocr/paddleocr/train')
#     # # img = cv2.imread(imagelist)
#     # img = randomErasing(imagelist)
#     # cv2.imshow('show', img)
#     # cv2.waitKey(0)
#     if __name__ == '__main__':
#         # imgDir = './0001.jpg'
#         # img = cv2.imread(imgDir)
#         path_name = r'112'
#         for item in os.listdir(path=path_name):
#             # print(item[-4:])
#             if item[-4:] == ".jpg":
#                 img_path = os.path.join(path_name, item)
#                 # print(img_path)
#                 img = randomErasing(img_path)
#                 cv2.imwrite('D:/img080803/' + item, img)
#                 # cv2.imwrite('D:/shiyan',img)
#                 # cv2.imshow('show', img)
#                 # cv2.waitKey(0)
import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = "stdlib"
import cv2
import sys, json
import time, re
import glob
import warnings
import configparser
from xml.dom.minidom import Document
from lxml import etree, objectify
from lxml.etree import Element
import traceback
import logging
import datetime
from tools.infer.predict import detect_roi


warnings.filterwarnings("ignore")
if not os.path.exists('barLog'):
    os.makedirs('barLog')
logfilename = f"barLog/Log{datetime.date.today().strftime('%Y%m%d')}.log"
logging.basicConfig(level=logging.INFO, filename=logfilename, format="%(asctime)s: %(module)s: %(message)s")


# img_path = sys.argv[1]
# ini_path = sys.argv[2]
# rotate = sys.argv[3]
# type_name = sys.argv[4]
# number_e = sys.argv[5]
img_path = './photo/12'
ini_path = 'photo/ini12'
rotate = '0'
type_name = '1'
number_e = '12'
#20版三相
fp = open('meter.json', encoding='utf-8')
meter_data = json.load(fp)
meter1 = meter_data["20版三相"]
meter2 = meter_data["20版单相"]
meter3 = meter_data["13版三相"]
meter4 = meter_data["13版单相"]
# 实例化一个xml节点
E = objectify.ElementMaker(annotate=False)

def detect_lcd(path, config, child1):
    # result_list = []
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.imread(path)

    # print('图片信息:', img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        gray = img.copy()
    # for i in range(1, len(cf.sections()) + 1):
    top_x = int(config.get(f"Location1", "topX"))
    top_y = int(config.get(f"Location1", "topY"))
    width = int(config.get(f"Location1", "width"))
    height = int(config.get(f"Location1", "height"))
    try:
        roi = img[top_y:top_y + height, top_x:top_x + width, ]
    except Exception as e:
        tree41 = E.BAR_CODE(f"{path} ini error")
        tree42 = E.CHECK_RESULT(
            E.REGION_TYPE("1"),
            E.CHECK_FLAG("NG")
        )
        child1.append(tree41)
        child1.append(tree42)
        return
    if rotate == '1':
        roi = cv2.flip(cv2.transpose(roi), 1)
    elif rotate == '2':
        roi = cv2.flip(roi, -1)
    elif rotate == '3':
        roi = cv2.flip(cv2.transpose(cv2.flip(roi, -1)), 1)
    cv2.imwrite("D:/378/",roi)

    # set_meter = []
    # results = detect_roi(roi)
    # print(results)
    # if results:
    #     # 识别结果转换为列表
    #     results_news = [res[0] for res in results]
    #     # for res in results:
    #     if type_name == '1':
    #         # 替换识别错误的字符
    #         # print(meter1[-1].split(','))
    #         for meter in meter1[-1].split(','):
    #             if meter:
    #                 temp = meter.split('>')
    #                 # print(temp)
    #                 results_news = [re.sub(temp[0],temp[1],res) for res in results_news]
    #         # print(results_news)
    #         for res in meter1[:-1]:
    #             if res not in results_news:
    #                 # print(res)
    #                 set_meter.append(res)
    #     elif type_name == '2':
    #         for meter in meter2[-1].split(','):
    #             if meter:
    #                 temp = meter.split('>')
    #                 results_news = [re.sub(temp[0], temp[1]) for res in results_news]
    #
    #         for res in meter2[:-1]:
    #             if res not in results_news:
    #                 set_meter.append(res)
    #     elif type_name == '3':
    #         for meter in meter3[-1].split(','):
    #             if meter:
    #                 temp = meter.split('>')
    #                 results_news = [re.sub(temp[0], temp[1]) for res in results_news]
    #
    #         for res in meter3[:-1]:
    #             if res not in results_news:
    #                 set_meter.append(res)
    #     elif type_name == '4':
    #         for meter in meter4[-1].split(','):
    #             if meter:
    #                 temp = meter.split('>')
    #                 results_news = [re.sub(temp[0], temp[1]) for res in results_news]
    #
    #         for res in meter4[:-1]:
    #             if res not in results_news:
    #                 set_meter.append(res)
    #     else:
    #         tree41 = E.BAR_CODE(f"{path} ini error")
    #         tree42 = E.CHECK_RESULT(
    #             E.REGION_TYPE("1"),
    #             E.CHECK_FLAG("NG, 请输入要识别的电表类型")
    #         )
    #         child1.append(tree41)
    #         child1.append(tree42)
    #         return
    #     # print(set_meter)
    #     if set_meter:
    #         tree41 = E.BAR_CODE(f"{path}")
    #         tree42 = E.CHECK_RESULT(
    #             E.REGION_TYPE("1"),
    #             E.CHECK_FLAG(f"NG,缺字{set_meter}")
    #         )
    #         child1.append(tree41)
    #         child1.append(tree42)
    #     else:
    #         tree41 = E.BAR_CODE(f"{path}")
    #         tree42 = E.CHECK_RESULT(
    #             E.REGION_TYPE("1"),
    #             E.CHECK_FLAG(f"OK")
    #         )
    #         child1.append(tree41)
    #         child1.append(tree42)
    #     cv2.imshow("3", roi)
    #     cv2.waitKey(1000)
    #     print(set_meter)
    #
    #     logging.info(f'接收参数：[图片路径：{path}，配置文件路径：{ini_path} ]\n识别结果：{results}\n')
    # else:
    #     tree41 = E.BAR_CODE(f"{path}")
    #     tree42 = E.CHECK_RESULT(
    #         E.REGION_TYPE("1"),
    #         E.CHECK_FLAG(f"NG,BLACK")
    #     )
    #     child1.append(tree41)
    #     child1.append(tree42)
    # logging.info(f'接收参数：[图片路径：{path}，配置文件路径：{ini_path} ]\n识别结果：{results}\n')


if __name__ == '__main__':

    start = time.time()
    root = Element("DATA")
    child1 = Element("METER")
    if not os.path.exists(img_path):
        # print('文件夹不存在')
        tree41 = E.BAR_CODE(f"{img_path}")
        tree42 = E.CHECK_RESULT(
            E.REGION_TYPE("1"),
            E.CHECK_FLAG(f"NG, {img_path}文件夹不存在，请输入包含图片的文件夹")
        )
        child1.append(tree41)
        child1.append(tree42)

    elif not os.path.exists(ini_path):
        # print('配置文件不存在')
        tree41 = E.BAR_CODE(f"{img_path}")
        tree42 = E.CHECK_RESULT(
            E.REGION_TYPE("1"),
            E.CHECK_FLAG(f"NG, {img_path}配置文件不存在，请输入正确配置文件路径")
        )
        child1.append(tree41)
        child1.append(tree42)
    else:
        imgs = glob.glob(img_path+'/*.jpg')  # 所有目录
        # 创建xml文件根节点
        for i in range(int(number_e)):
            # print(f'第{i + 1}个镜头')
            img_path_temp = img_path + f'/PICTURE{i + 1}\*.jpg'
            ini = ini_path + f'/config{i+1}.ini'
            imgs = glob.glob(img_path_temp)  # 所有图片
            # if range_d.find('1') != -1:
            # 显示屏模板
            cf = configparser.ConfigParser()
            cf.read(ini)
            # print(os.path.exists(ini_path),ini)
            for img in imgs:
                detect_lcd(img,cf,child1)
    root.append(child1)
    result = etree.ElementTree(root)
    result.write('result.xml', pretty_print=True, xml_declaration=True, encoding='utf-8')
    # except:
    #     s = traceback.format_exc()
    #     logging.error(s)
    #     doc = Document()
    #     nodeError = doc.createElement('ERROR')
    #     doc.appendChild(nodeError)
    #     nodeError.appendChild(doc.createTextNode(f"An unknown error has occurred, see details in {logfilename}"))
    #     fp = open('result.xml', 'w')
    #     doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='UTF-8')
    #     fp.close()
    # print('总耗时',time.time()-start)
