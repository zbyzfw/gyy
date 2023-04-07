# coding: utf-8
import sys
import cv2
import numpy as np
import glob
from pyzbar.pyzbar import decode
from lxml import etree, objectify


"""
第一个为存放图片的文件夹，第二个为存放模板的文件夹，第三个为检测显示屏的阈值和以前一样为3，第四个参数决定是否翻转图片，一般为0（表示不翻转），
第五个参数为检测项参数，1代表检测显示屏，2代表检测条形码，3代表检测跳闸灯，4代表检测外观铭牌,如果只检测显示屏则为1，如果检测显示屏与外观则为14，
如果四项全部检测则为1234；第六个参数为表位数（1-6）。
# """
img_path = "img37"
template_path ="model37"
threshold = '5'
rotate = '0'
range_d = '1'
number_e = '4'
# img_path = sys.argv[1]  #返回命令行参数
# template_path = sys.argv[2]
# threshold = sys.argv[3]
# rotate = sys.argv[4]
# range_d = sys.argv[5]#第5个参数
# number_e = sys.argv[6]#返回索引第6个参数


def detect_biaowei(flag0, imgs, rotate, range_d, threshold, model_txt1, model_txt2, gylb, template1, template2, E, result_xml):
    flag1 = 1
    flag2 = 1
    flag3 = 1
    flag4 = 1
    judge_type = 1

    if len(template1.shape) == 3:  # 彩色还是黑白
        template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)#转为灰度图
    else:
        template1 = template1.copy()
    # 模板宽高
    w2, h2 = template1.shape[::-1]
    if range_d.find('4') != -1:
        if len(template2.shape) == 3:
            template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
        else:
            template2 = template2.copy()
        # 图像宽高
        #print(template2.shape[::-1])
        w4, h4 = template2.shape[::-1]
    # cv2.imshow("img1",cv2.resize(template2,(600,600)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 主要代码位置
    for img_p in imgs:
        img = cv2.imdecode(np.fromfile(img_p, dtype=np.uint8), -1)
        if rotate == '1':
            img = cv2.flip(cv2.transpose(img), 1)
        elif rotate == '2':
            img = cv2.flip(img, -1)
        elif rotate == '3':
            img = cv2.flip(cv2.transpose(cv2.flip(img, -1)), 1)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        # cv2.imshow("img1",cv2.resize(gray,(600,600)))
        # cv2.waitKey(0)
        ##------------------------以下代码矫正图片歪斜情况---------------------------------------------------
        w1, h1 = img.shape[:2]
        center = (h1 // 2, w1 // 2)  # 中值
        #print(center)
        # 根据高度判断l值,l值越大直线越细致
        if h2 > 700:
            l = 0.2
        elif h2 > 500:
            l = 0.5
        elif h2 > 300:
            l = 0.7
        else:
            l = 0.8

        # 高斯模糊
        gray0 = cv2.GaussianBlur(gray, (9, 9), 0)
        # gray0 = cv2.resize(gray0,(600,600))
        # cv2.imshow("img1", gray0)
        # cv2.waitKey(0)
        # 参数(0不加强细化, 用于查找直线的图像比例)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, l)  # 检测直线
        lines = lsd.detect(gray0)
        # with_line_img = lsd.drawSegments(gray0, lines[0])
        # # cv2.imshow("img1",cv2.resize(with_line_img,(600,600)))
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        #print(lines[0])
        if lines[0] is not None:
            # 遍历每条检测出的线
            for line in lines[0]:
                # 获取合适长度的线(line1-line3绝对值 > 4*模板高度/5)
                if np.abs(line[0][0] - line[0][2]) > (4 * w2 / 5) \
                        and np.abs(line[0][1] - line[0][3]) < (h2 / 10):
                    # if int((line[0][2] - line[0][1]))>200:
                    line2 = line
                    # print(line2)
                    # 判断获取到的线是否准确
                    if line2[0][3] != line2[0][1] and line2[0][2] != line2[0][0]:
                        break
            try:
                # 计算角度 对边/临边 算出角度后乘180/pi
                angle = np.arctan(float(line2[0][3] - line2[0][1]) /
                                  float(line2[0][2] - line2[0][0])) * 180 / np.pi

                # 将图像以一定角度旋转,获得转换矩阵
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # 仿射变换将图像矫正对齐,宽高和原来一致
                rotated = cv2.warpAffine(img, M, (h1, w1))
                # 灰度变换
                if len(rotated.shape) == 3:
                    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                else:
                    gray2 = rotated.copy()
            except Exception as e:
                #print(e)
                rotated = img
                if len(rotated.shape) == 3:
                    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

                else:
                    gray2 = rotated.copy()
            # cv2.imshow("img1", cv2.resize(gray2, (900, 900)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    #---------------------------------模板匹配-----------------------------------------------------------

            _, max_val, _, top_left2 = cv2.minMaxLoc(cv2.matchTemplate
                                               (gray2, template1, cv2.TM_CCOEFF_NORMED))#模板匹配得到最大值以及最大值的索引确定位置
            # binary_img = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # cv2.imshow('img', cv2.resize(max_val,(800,900)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print('电表匹配结果: ', max_val)
            # bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
            # # print(top_left2,"\n",bottom_right2,rotated.shape)
            #
            # roi = rotated[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]]
            # roi1 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            # # cv2.imshow('roi', cv2.resize(roi,(800,900)))
            # # cv2.waitKey(0)
            judge1 = True
            judge2 = True
    ##==========================================判断匹配不准确的情况=========================================================
            if max_val < 0.65:
                # 由模板匹配结果截取led
                bottom_right3 = (top_left2[0] + w2, top_left2[1] + h2)
                roi1 = gray2[top_left2[1]-20:bottom_right3[1]+20, top_left2[0]-10:bottom_right3[0]+20]
                binary_img = cv2.threshold(roi1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]  # 二值化
                # cv2.imshow('img', binary_img)
                # cv2.waitKey(0)
                # 宽度不准确导致轮廓检测出错
                # cv.RETR_TREE会完整建立轮廓的层级从属关系，[1]代表只获得hierarchy，它是一个包含4个值的数组：[Next, Previous, First Child, Parent]
                contours1 = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                # print(contours1)
                # 对轮廓的面积降序排列
                contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
                for error_value in contours1:
                    minArea = cv2.minAreaRect(error_value)
                    area = cv2.contourArea(error_value)
                    # 面积大于1.05的轮廓进行下一步判断
                    if area/(w2 * h2) > 0.92:
                    # if area / (w2 * h2) > 1.05:
                        judge2 = False
                        continue
                    # print('面积',area/(w2 * h2))
                    roi2 = cv2.drawContours(roi1,error_value,-1,(255,0,0),5)
                    # cv2.imshow('roi2', roi2)
                    # cv2.waitKey(1000000)
                    if minArea[1][0] > minArea[1][1]:
                        width = int(minArea[1][0])
                        height = int(minArea[1][1])
                    else:
                        width = int(minArea[1][1])
                        height = int(minArea[1][0])
                    # 判断轮廓的大小是否合适
                    # print(width,height)
                    # print(w2,h2)
                    # if w2-100 < width < w2+40 and h2-100 < height < h2+40 :
                    if w2 - 50 < width < w2 + 40 and h2 - 50 < height < h2 + 40 and area / (w2 * h2) > 0.85:
                        judge1 = False
                        break
                print(judge1, judge2, flag1)
                if not judge2 and judge1:#不同时为真
                    judge1 = False
                if judge1:
                    flag0 = flag0 + 1
                    bars = str(img_p)
                    flag1 = 0
                    flag3 = 0
                    flag4 = 0
                    if flag0 == 1:
                        E = objectify.ElementMaker(annotate=False)
                        result_xml = E.DATA(
                            E.METER(
                                E.BAR_CODE(bars),
                                E.CHECK_RESULT(
                                    E.REGION_TYPE("1"),
                                    E.CHECK_FLAG(flag1)
                                ),
                                E.CHECK_RESULT(
                                    E.REGION_TYPE("3"),
                                    E.CHECK_FLAG(flag3)
                                ),
                                E.CHECK_RESULT(
                                    E.REGION_TYPE("4"),
                                    E.CHECK_FLAG(flag4)
                                ),
                            ),
                        )
                        etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
                        flag1 = 1
                        flag3 = 1
                        flag4 = 1
                    else:
                        E1 = objectify.ElementMaker(annotate=False)
                        result_xml1 = E1.METER(
                            E.BAR_CODE(bars),
                            E.CHECK_RESULT(
                                E.REGION_TYPE("1"),
                                E.CHECK_FLAG(flag1)
                            ),
                            E.CHECK_RESULT(
                                E.REGION_TYPE("3"),
                                E.CHECK_FLAG(flag3)
                            ),
                            E.CHECK_RESULT(
                                E.REGION_TYPE("4"),
                                E.CHECK_FLAG(flag4)
                            ),
                        )
                        result_xml.append(result_xml1)
                        etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
                        flag1 = 1
                        flag3 = 1
                        flag4 = 1
                    continue

            # 新增代码+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if len(rotated.shape) == 3:
                rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            else:
                rotated = rotated.copy()

            # 由模板匹配结果得到图像坐标w2
            bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
            #print(top_left2,"\n",bottom_right2,rotated.shape)

            roi = rotated[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]]
            roi1 = cv2.cvtColor(roi,cv2.COLOR_GRAY2BGR)
            # cv2.imshow('roi', cv2.resize(roi,(800,900)))
            # cv2.waitKey(0)
            if len(roi.shape) == 3:
                dst = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                dst = roi.copy()

            # 判断是否检测显示屏
            if range_d.find('1') != -1:
                str1 = ''
                error = 0
                with open(model_txt1, 'r') as file2:
                    lines = file2.readlines()  # 获取位置信息
                    # print(len(lines))
                    if len(lines) > 210:
                        judge_type = 2

                    error_value = False
                    char_exist = False
                    line = lines[0]
                    #print(line)
                    list = line.split()
                    #print(list)
                    if w2 < 1300:
                        error_range = 6
                    else:
                        error_range = 10
                    rect = roi[int(list[1]) - (int(list[3])-int(list[1]))//2:int(list[3]) + (int(list[3])-int(list[1]))//2,
                               int(list[0]) - (int(list[2])-int(list[0]))//2:int(list[2]) + (int(list[2])-int(list[0]))//2]
                    # rect = roi[int(list[1]) - 50:int(list[3]) + 60, int(list[0]) - 30:int(list[2]) + 35]
                    # cv2.imshow("111",rect)
                    # cv2.waitKey(1)
                    if len(rect.shape) == 3:
                        rect_gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
                    else:
                        rect_gray = rect.copy()

                    ret3, th3 = cv2.threshold(rect_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # _, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    # cv2.imshow('rect', rect_gray)
                    # cv2.waitKey(1000)
                    # cv2.destroyAllWindows()
                    for i in range(len(contours)):
                        # if i > 0:
                        st_x, st_y, width, height = cv2.boundingRect(contours[i])
                        c_x = st_x + width // 2
                        if (int(list[3])-int(list[1])) - error_range < height < int(list[3])-int(list[1])+error_range and int(list[2])-int(list[0])-error_range < width < int(list[2])-int(list[0])+error_range:
                            c_x1 = c_x
                            c_y1 = st_y + height // 2
                            char_exist = True
                            bound_rect = np.array([[[st_x, st_y]], [[st_x + width, st_y]],
                                                   [[st_x + width, st_y + height]],
                                                   [[st_x, st_y + height]]])
                            cv2.drawContours(rect_gray, [bound_rect], -1, (0, 255, 255), 1)

                            break
                    if char_exist:
                        # rect_x = int(list[0]) - 30 + c_x1
                        # rect_y = int(list[1]) - 50 + c_y1
                        rect_x = int(list[0]) - (int(list[2])-int(list[0]))//2 + c_x1
                        rect_y = int(list[1]) - (int(list[3])-int(list[1]))//2 + c_y1
                        center_x = int(list[0]) + (int(list[2]) - int(list[0])) // 2
                        center_y = int(list[1]) + (int(list[3]) - int(list[1])) // 2
                        move_x = int(rect_x) - int(center_x)
                        move_y = int(rect_y) - int(center_y)
                        # print(move_x)
                        error_value = True
                    # cv2.imshow("rect", rect_gray)
                    # cv2.waitKey(0)
                    for line in lines:
                        list = line.split()
                        if error_value:
                            std = np.std(dst[max(0, int(list[1])+move_y):int(list[3])+move_y, int(list[0]) + move_x:int(list[2]) + move_x])#计算图片某值标准差
                        else:
                            std = np.std(dst[int(list[1]):int(list[3]), int(list[0]):int(list[2])])
                        print(list[4] + ' std:' + str(std))
                        if std < float(threshold):#标准差小于阈值
                            print(list[4]+' std:'+str(std))
                            if error_value:
                                cv2.rectangle(roi1, (int(list[0]) + move_x, max(0, int(list[1]) + move_y)),
                                              (int(list[2]) + move_x, int(list[3]) + move_y), (0, 0, 255), 1)
                                error = error + 1
                                str1 = str1 + ' ' + str(list[4])
                            else:
                                cv2.rectangle(roi1, (int(list[0]), int(list[1])),
                                              (int(list[2]), int(list[3])), (0, 0, 255), 1)
                                error = error + 1
                                str1 = str1 + ' ' + str(list[4])
                        else:
                            if error_value:
                                cv2.rectangle(roi1, (int(list[0]) + move_x, int(list[1]) + move_y),
                                              (int(list[2]) + move_x, int(list[3]) + move_y), (0, 255, 0), 1)
                            else:
                                cv2.rectangle(roi1, (int(list[0]), int(list[1])),
                                              (int(list[2]), int(list[3])), (0, 255, 0), 1)
                    # cv2.imshow('img', roi1)
                    # cv2.waitKey(0)
                # 没有错误时,flag置为1,出现错误时flag为错误类型
                if error == 0:
                    flag1 = 1
                else:
                    flag1 = 0
            # 是否检测外观铭牌
            if range_d.find('4') != -1:
                # 新增代码适配显示器模板标注较少的情况=================================================================
                # with open(model_txt1, 'r') as file2:
                #     lines = file2.readlines()
                    # if len(lines) > 220:
                    #     cropped_test1 = gray2[top_left2[1]:bottom_right2[1], bottom_right2[0]:w1]
                    # else:
                    #     cropped_test1 = gray2[top_left2[1]:bottom_right2[1], bottom_right2[0]:w1]
                # 原代码===================================================================================
                if judge_type == 1:
                    cropped_test1 = gray2[0:int(top_left2[1]), int(top_left2[0]):w1]
                else:
                    # 修改后的代码bottom
                    cropped_test1 = gray2[top_left2[1]:bottom_right2[1], bottom_right2[0]-20:w1]
                # cropped_test1 =cv2.resize(cropped_test1,(600,600))
                # cv2.imshow('gy', cropped_test1)
                # cv2.waitKey(100000)
                cropped_model1 = cv2.imdecode(np.fromfile(gylb, dtype=np.uint8), -1)
                # cv2.imshow('gy', cropped_model1)

                # cv2.waitKey(1000)
                if len(cropped_model1.shape) == 3:
                    cropped_model1 = cv2.cvtColor(cropped_model1, cv2.COLOR_BGR2GRAY)
                else:
                    cropped_model1 = cropped_model1.copy()
                cropped_model1 = cv2.GaussianBlur(cropped_model1, (9, 9), 0)

                _, maxres1, _, _ = cv2.minMaxLoc(cv2.matchTemplate
                                                 (cropped_model1, cropped_test1, cv2.TM_CCOEFF_NORMED))#模板匹配得到得分
                # print(maxres1)
                if maxres1 < 0.6:
                    flag4 = 0
            # 是否检测跳闸灯
            if range_d.find('3') != -1:
                if judge_type == 1:
                    light = gray[int(bottom_right2[1]) + int((bottom_right2[1] - top_left2[1]) / 7):int(
                        bottom_right2[1] + int((bottom_right2[1] - top_left2[1]) / 2)),
                            int(top_left2[0]) + int((bottom_right2[0] - top_left2[0]) / 4):int(top_left2[0]) + int(
                                (bottom_right2[0] - top_left2[0]) / 2)]
                else:
                    light = gray[int(bottom_right2[1]):int(bottom_right2[1] + int((bottom_right2[1] - top_left2[1]) * 2 / 5)),
                            int(top_left2[0]) + int((bottom_right2[0] - top_left2[0]) * 5 / 12):int(top_left2[0]) + int(
                                (bottom_right2[0] - top_left2[0]) * 7 / 12)]
                # cv2.imshow("light",light)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                h5, w5 = light.shape[:2]
                # 霍夫变换找圆
                circles1 = cv2.HoughCircles(light, cv2.HOUGH_GRADIENT, 1, 100, param1=200, param2=30, minRadius=0,
                                            maxRadius=0)
                # print('霍夫变换结果',circles1)
                if circles1 is not None:
                    circles = circles1[0, :, :]
                    circles = np.uint16(np.around(circles))
                    # print(circles[0][0])
                    # print(circles[0][1])
                    x = circles[0][0]
                    y = circles[0][1]
                    r = circles[0][2]
                    # print('霍夫变换结果:圆心坐标及半径', x, y, r)
                    img0 = light[y - r:y + r, x - r:x + r]
                    # img0 = light[20:110, 20:75]
                    th_img0 = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    h6, w6 = img0.shape[:2]
                    # cv2.imwrite('led.jpg', img0)
                    # if img0:
                    # cv2.imshow('result', img0)
                    # cv2.waitKey(1000)
                    th = np.sum(th_img0 == 0)
                    # print(th > (w6 * h6 / 2))
                    if th > (w6 * h6 / 2):
                        flag3 = 0
                    else:
                        flag3 = 1
                else:
                    th = np.sum(light < 40)
                    if th > (w5 * h5 / 2):
                        flag3 = 0
                    else:
                        flag3 = 1
                # print(flag3)
            if range_d.find('4') != -1 and flag4 == 1:
                _, _, _, top_left2 = cv2.minMaxLoc(cv2.matchTemplate
                                                   (gray2, template2, cv2.TM_CCOEFF_NORMED))#模板匹配得到最大得分位置

                bottom_right2 = (top_left2[0] + w4, top_left2[1] + h4)
                roi = rotated[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]]
                # cv2.imshow('roi',cv2.resize(roi,(600,600)))
                # cv2.waitKey(0)
                if len(roi.shape) == 3:
                    dst = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    dst = roi.copy()
                with open(model_txt2, 'r') as file2:
                    lines = file2.readlines()
                    i = 1
                    for line in lines:
                        list = line.split()
                        cropped_model = template2[int(list[1]):int(list[3]), int(list[0]):int(list[2])]
                        if int(list[1]) - 40 < 0:
                            list[1] = 0
                        else:
                            list[1] = int(list[1]) - 40

                        if int(list[3]) + 40 > h4:
                            list[3] = h4
                        else:
                            list[3] = int(list[3]) + 40

                        if int(list[0]) - 40 < 0:
                            list[0] = 0
                        else:
                            list[0] = int(list[0]) - 40

                        if int(list[2]) + 40 > w4:
                            list[2] = w4
                        else:
                            list[2] = int(list[2]) + 40

                        cropped_test = dst[int(list[1]):int(list[3]), int(list[0]):int(list[2])]
                        cropped_model = cv2.GaussianBlur(cropped_model, (9, 9), 0)
                        cropped_test = cv2.GaussianBlur(cropped_test, (9, 9), 0)
                        # cv2.imshow('model',cropped_model)
                        # cv2.waitKey(1000)
                        # cv2.imshow('test',cropped_test)
                        # cv2.waitKey(10000)

                        _, maxres, _, _ = cv2.minMaxLoc(cv2.matchTemplate
                                                        (cropped_test, cropped_model, cv2.TM_CCOEFF_NORMED))#模板匹配
                        # print('111',maxres)
                        if line == lines[0]:
                            if maxres < 0.6:
                                flag4 = 0
                                break

                        elif line != lines[len(lines) - 1] and line != lines[0]:
                            i = i + 1
                            if maxres < 0.6:
                                flag4 = 0
                                break

                        elif line == lines[len(lines) - 1]:
                            if maxres < 0.6:
                                flag4 = 0
                            else:
                                flag4 = 1
                # bars = 0
            # 需要识别条码
            if range_d.find('2') != -1:
                # cv2.imshow('bars:',cv2.resize(rotated[int(w1 / 2):int(w1), int(h1 / 5):int(h1)],(400,300)))
                # cv2.waitKey(0)
                bars = decode(rotated[int(w1 / 2):int(w1), int(h1 / 5):int(h1)])
                # print('识别到的条形码: ', bars)

                if bars:
                    flag2 = 1
                    # 遍历所有识别到的二维码
                    for barcode in bars:
                        bars = barcode.data.decode('utf-8')
                        break
                else:
                    flag2 = 0
                    bars = str(img_p)
            else:
                # 不检测条码时bars为图片名
                bars = str(img_p)
        else:
            flag1 = 0
            flag3 = 0
            flag4 = 0
        flag0 = flag0 + 1

        if flag0 == 1:
            E = objectify.ElementMaker(annotate=False)
            result_xml = E.DATA(
                E.METER(
                    E.BAR_CODE(bars),
                    E.CHECK_RESULT(
                        E.REGION_TYPE("1"),
                        E.CHECK_FLAG(flag1)
                    ),
                    E.CHECK_RESULT(
                        E.REGION_TYPE("3"),
                        E.CHECK_FLAG(flag3)
                    ),
                    E.CHECK_RESULT(
                        E.REGION_TYPE("4"),
                        E.CHECK_FLAG(flag4)
                    ),
                ),
            )
            etree.ElementTree(result_xml).write("result.xml", pretty_print=True)

        else:
            E1 = objectify.ElementMaker(annotate=False)
            result_xml1 = E1.METER(
                E.BAR_CODE(bars),
                E.CHECK_RESULT(
                    E.REGION_TYPE("1"),
                    E.CHECK_FLAG(flag1)
                ),
                E.CHECK_RESULT(
                    E.REGION_TYPE("3"),
                    E.CHECK_FLAG(flag3)
                ),
                E.CHECK_RESULT(
                    E.REGION_TYPE("4"),
                    E.CHECK_FLAG(flag4)
                ),
            )
            result_xml.append(result_xml1)
            etree.ElementTree(result_xml).write("result.xml", pretty_print=True)

    return E, result_xml#返回参数


def run(img_path,template_path,threshold,rotate,range_d,number_e):#定义run函数
      ##------------------------获取路径--------------------------------------------------------------------
    try:
        if number_e == '1':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            imgs1 = glob.glob(img_path1)  # 所有目录
            #print(imgs1)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)#从网络传输中恢复图像
            if range_d.find('4') != -1:
                model_exterior1 = template_path + '\B1.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                gylb1 = template_path + '\gy1.jpg'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)#从网络传输中恢复图像
            else:
                template_exterior1 = ""
                gylb1 = ""
                exterior_txt1 = ""
        elif number_e == '2':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            img_path2 = img_path + '/PICTURE2\*.jpg'
            imgs1 = glob.glob(img_path1)
            imgs2 = glob.glob(img_path2)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                model_lcd2 = template_path + '\A2.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                lcd_txt2 = template_path + '\A2.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)
                template_lcd2 = cv2.imdecode(np.fromfile(model_lcd2, dtype=np.uint8), -1)
            if range_d.find('4') != -1:
                model_exterior1 = template_path + '\B1.jpg'
                model_exterior2 = template_path + '\B2.jpg'
                gylb1 = template_path + '\gy1.jpg'
                gylb2 = template_path + '\gy2.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                exterior_txt2 = template_path + '\B2.txt'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)
                template_exterior2 = cv2.imdecode(np.fromfile(model_exterior2, dtype=np.uint8), -1)
            else:
                template_exterior1 = ""
                template_exterior2 = ""
                gylb1 = ""
                gylb2 = ""
                exterior_txt1 = ""
                exterior_txt2 = ""
        elif number_e == '3':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            img_path2 = img_path + '/PICTURE2\*.jpg'
            img_path3 = img_path + '/PICTURE3\*.jpg'
            imgs1 = glob.glob(img_path1)
            imgs2 = glob.glob(img_path2)
            imgs3 = glob.glob(img_path3)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                model_lcd2 = template_path + '\A2.jpg'
                model_lcd3 = template_path + '\A3.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                lcd_txt2 = template_path + '\A2.txt'
                lcd_txt3 = template_path + '\A3.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)
                template_lcd2 = cv2.imdecode(np.fromfile(model_lcd2, dtype=np.uint8), -1)
                template_lcd3 = cv2.imdecode(np.fromfile(model_lcd3, dtype=np.uint8), -1)
            if range_d.find('4') != -1:
                model_exterior3 = template_path + '\B3.jpg'
                model_exterior1 = template_path + '\B1.jpg'
                model_exterior2 = template_path + '\B2.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                exterior_txt2 = template_path + '\B2.txt'
                exterior_txt3 = template_path + '\B3.txt'
                gylb1 = template_path + '\gy1.jpg'
                gylb2 = template_path + '\gy2.jpg'
                gylb3 = template_path + '\gy3.jpg'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)
                template_exterior2 = cv2.imdecode(np.fromfile(model_exterior2, dtype=np.uint8), -1)
                template_exterior3 = cv2.imdecode(np.fromfile(model_exterior3, dtype=np.uint8), -1)
            else:
                template_exterior1 = ""
                template_exterior2 = ""
                template_exterior3 = ""
                gylb1 = ""
                gylb2 = ""
                gylb3 = ""
                exterior_txt1 = ""
                exterior_txt2 = ""
                exterior_txt3 = ""
        elif number_e == '4':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            img_path2 = img_path + '/PICTURE2\*.jpg'
            img_path3 = img_path + '/PICTURE3\*.jpg'
            img_path4 = img_path + '/PICTURE4\*.jpg'
            imgs1 = glob.glob(img_path1)#得到所有图片
            imgs2 = glob.glob(img_path2)
            imgs3 = glob.glob(img_path3)
            imgs4 = glob.glob(img_path4)
            #print(imgs1,imgs2,imgs3,imgs4)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                model_lcd2 = template_path + '\A2.jpg'
                model_lcd3 = template_path + '\A3.jpg'
                model_lcd4 = template_path + '\A4.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                lcd_txt2 = template_path + '\A2.txt'
                lcd_txt3 = template_path + '\A3.txt'
                lcd_txt4 = template_path + '\A4.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)
                template_lcd2 = cv2.imdecode(np.fromfile(model_lcd2, dtype=np.uint8), -1)
                template_lcd3 = cv2.imdecode(np.fromfile(model_lcd3, dtype=np.uint8), -1)
                template_lcd4 = cv2.imdecode(np.fromfile(model_lcd4, dtype=np.uint8), -1)
            if range_d.find('4') != -1:
                model_exterior1 = template_path + '\B1.jpg'
                model_exterior2 = template_path + '\B2.jpg'
                model_exterior3 = template_path + '\B3.jpg'
                model_exterior4 = template_path + '\B4.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                exterior_txt2 = template_path + '\B2.txt'
                exterior_txt3 = template_path + '\B3.txt'
                exterior_txt4 = template_path + '\B4.txt'
                gylb1 = template_path + '\gy1.jpg'
                gylb2 = template_path + '\gy2.jpg'
                gylb3 = template_path + '\gy3.jpg'
                gylb4 = template_path + '\gy4.jpg'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)
                template_exterior2 = cv2.imdecode(np.fromfile(model_exterior2, dtype=np.uint8), -1)
                template_exterior3 = cv2.imdecode(np.fromfile(model_exterior3, dtype=np.uint8), -1)
                template_exterior4 = cv2.imdecode(np.fromfile(model_exterior4, dtype=np.uint8), -1)
            else:
                template_exterior1 = ""
                template_exterior2 = ""
                template_exterior3 = ""
                template_exterior4 = ""
                gylb1 = ""
                gylb2 = ""
                gylb3 = ""
                gylb4 = ""
                exterior_txt1 = ""
                exterior_txt2 = ""
                exterior_txt3 = ""
                exterior_txt4 = ""
        elif number_e == '5':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            img_path2 = img_path + '/PICTURE2\*.jpg'
            img_path3 = img_path + '/PICTURE3\*.jpg'
            img_path4 = img_path + '/PICTURE4\*.jpg'
            img_path5 = img_path + '/PICTURE5\*.jpg'
            imgs1 = glob.glob(img_path1)
            imgs2 = glob.glob(img_path2)
            imgs3 = glob.glob(img_path3)
            imgs4 = glob.glob(img_path4)
            imgs5 = glob.glob(img_path5)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                model_lcd2 = template_path + '\A2.jpg'
                model_lcd3 = template_path + '\A3.jpg'
                model_lcd4 = template_path + '\A4.jpg'
                model_lcd5 = template_path + '\A5.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                lcd_txt2 = template_path + '\A2.txt'
                lcd_txt3 = template_path + '\A3.txt'
                lcd_txt4 = template_path + '\A4.txt'
                lcd_txt5 = template_path + '\A5.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)
                template_lcd2 = cv2.imdecode(np.fromfile(model_lcd2, dtype=np.uint8), -1)
                template_lcd3 = cv2.imdecode(np.fromfile(model_lcd3, dtype=np.uint8), -1)
                template_lcd4 = cv2.imdecode(np.fromfile(model_lcd4, dtype=np.uint8), -1)
                template_lcd5 = cv2.imdecode(np.fromfile(model_lcd5, dtype=np.uint8), -1)
            if range_d.find('4') != -1:
                model_exterior1 = template_path + '\B1.jpg'
                model_exterior2 = template_path + '\B2.jpg'
                model_exterior3 = template_path + '\B3.jpg'
                model_exterior4 = template_path + '\B4.jpg'
                model_exterior5 = template_path + '\B5.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                exterior_txt2 = template_path + '\B2.txt'
                exterior_txt3 = template_path + '\B3.txt'
                exterior_txt4 = template_path + '\B4.txt'
                exterior_txt5 = template_path + '\B5.txt'
                gylb1 = template_path + '\gy1.jpg'
                gylb2 = template_path + '\gy2.jpg'
                gylb3 = template_path + '\gy3.jpg'
                gylb4 = template_path + '\gy4.jpg'
                gylb5 = template_path + '\gy5.jpg'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)
                template_exterior2 = cv2.imdecode(np.fromfile(model_exterior2, dtype=np.uint8), -1)
                template_exterior3 = cv2.imdecode(np.fromfile(model_exterior3, dtype=np.uint8), -1)
                template_exterior4 = cv2.imdecode(np.fromfile(model_exterior4, dtype=np.uint8), -1)
                template_exterior5 = cv2.imdecode(np.fromfile(model_exterior5, dtype=np.uint8), -1)
            else:
                template_exterior1 = ""
                template_exterior2 = ""
                template_exterior3 = ""
                template_exterior4 = ""
                template_exterior5 = ""
                gylb1 = ""
                gylb2 = ""
                gylb3 = ""
                gylb4 = ""
                gylb5 = ""
                exterior_txt1 = ""
                exterior_txt2 = ""
                exterior_txt3 = ""
                exterior_txt4 = ""
                exterior_txt5 = ""
        elif number_e == '6':
            img_path1 = img_path + '/PICTURE1\*.jpg'
            img_path2 = img_path + '/PICTURE2\*.jpg'
            img_path3 = img_path + '/PICTURE3\*.jpg'
            img_path4 = img_path + '/PICTURE4\*.jpg'
            img_path5 = img_path + '/PICTURE5\*.jpg'
            img_path6 = img_path + '/PICTURE6\*.jpg'
            imgs1 = glob.glob(img_path1)
            imgs2 = glob.glob(img_path2)
            imgs3 = glob.glob(img_path3)
            imgs4 = glob.glob(img_path4)
            imgs5 = glob.glob(img_path5)
            imgs6 = glob.glob(img_path6)
            if range_d.find('1') != -1:
                model_lcd1 = template_path + '\A1.jpg'
                model_lcd2 = template_path + '\A2.jpg'
                model_lcd3 = template_path + '\A3.jpg'
                model_lcd4 = template_path + '\A4.jpg'
                model_lcd5 = template_path + '\A5.jpg'
                model_lcd6 = template_path + '\A6.jpg'
                lcd_txt1 = template_path + '\A1.txt'
                lcd_txt2 = template_path + '\A2.txt'
                lcd_txt3 = template_path + '\A3.txt'
                lcd_txt4 = template_path + '\A4.txt'
                lcd_txt5 = template_path + '\A5.txt'
                lcd_txt6 = template_path + '\A6.txt'
                template_lcd1 = cv2.imdecode(np.fromfile(model_lcd1, dtype=np.uint8), -1)
                template_lcd2 = cv2.imdecode(np.fromfile(model_lcd2, dtype=np.uint8), -1)
                template_lcd3 = cv2.imdecode(np.fromfile(model_lcd3, dtype=np.uint8), -1)
                template_lcd4 = cv2.imdecode(np.fromfile(model_lcd4, dtype=np.uint8), -1)
                template_lcd5 = cv2.imdecode(np.fromfile(model_lcd5, dtype=np.uint8), -1)
                template_lcd6 = cv2.imdecode(np.fromfile(model_lcd6, dtype=np.uint8), -1)
            if range_d.find('4') != -1:
                model_exterior1 = template_path + '\B1.jpg'
                model_exterior2 = template_path + '\B2.jpg'
                model_exterior3 = template_path + '\B3.jpg'
                model_exterior4 = template_path + '\B4.jpg'
                model_exterior5 = template_path + '\B5.jpg'
                model_exterior6 = template_path + '\B6.jpg'
                exterior_txt1 = template_path + '\B1.txt'
                exterior_txt2 = template_path + '\B2.txt'
                exterior_txt3 = template_path + '\B3.txt'
                exterior_txt4 = template_path + '\B4.txt'
                exterior_txt5 = template_path + '\B5.txt'
                exterior_txt6 = template_path + '\B6.txt'
                gylb1 = template_path + '\gy1.jpg'
                gylb2 = template_path + '\gy2.jpg'
                gylb3 = template_path + '\gy3.jpg'
                gylb4 = template_path + '\gy4.jpg'
                gylb5 = template_path + '\gy5.jpg'
                gylb6 = template_path + '\gy6.jpg'
                template_exterior1 = cv2.imdecode(np.fromfile(model_exterior1, dtype=np.uint8), -1)
                template_exterior2 = cv2.imdecode(np.fromfile(model_exterior2, dtype=np.uint8), -1)
                template_exterior3 = cv2.imdecode(np.fromfile(model_exterior3, dtype=np.uint8), -1)
                template_exterior4 = cv2.imdecode(np.fromfile(model_exterior4, dtype=np.uint8), -1)
                template_exterior5 = cv2.imdecode(np.fromfile(model_exterior5, dtype=np.uint8), -1)
                template_exterior6 = cv2.imdecode(np.fromfile(model_exterior6, dtype=np.uint8), -1)
            else:
                template_exterior1 = ""
                template_exterior2 = ""
                template_exterior3 = ""
                template_exterior4 = ""
                template_exterior5 = ""
                template_exterior6 = ""
                gylb1 = ""
                gylb2 = ""
                gylb3 = ""
                gylb4 = ""
                gylb5 = ""
                gylb6 = ""
                exterior_txt1 = ""
                exterior_txt2 = ""
                exterior_txt3 = ""
                exterior_txt4 = ""
                exterior_txt5 = ""
                exterior_txt6 = ""
        else:
           print('biaowei find error')
    except:
        E = objectify.ElementMaker(annotate=False)
        result_xml = E.DATA(
            E.METER(
                E.CHECK_RESULT(
                    E.REGION_TYPE("template search error"),
                ),
            ),
        )
        etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
    ##----------------------调用detect_biaowei获取结果-----------------------------------------------------
    try:
        if number_e == '1':
            detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1, template_lcd1,
                                   template_exterior1, None, None)

        elif number_e == '2':

            E, result_xml = detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1,
                                           template_lcd1, template_exterior1, None, None)
            detect_biaowei(1, imgs2, rotate, range_d, threshold, lcd_txt2, exterior_txt2, gylb2, template_lcd2,
                           template_exterior2, E, result_xml)

        elif number_e == '3':

            E, result_xml = detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1,
                                           template_lcd1, template_exterior1, None, None)
            E, result_xml = detect_biaowei(1, imgs2, rotate, range_d, threshold, lcd_txt2, exterior_txt2, gylb2,
                                           template_lcd2, template_exterior2, E, result_xml)
            detect_biaowei(1, imgs3, rotate, range_d, threshold, lcd_txt3, exterior_txt3, gylb3, template_lcd3,
                           template_exterior3, E, result_xml)

        elif number_e == '4':

            E, result_xml = detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1,
                                           template_lcd1, template_exterior1, None, None)
            E, result_xml = detect_biaowei(1, imgs2, rotate, range_d, threshold, lcd_txt2, exterior_txt2, gylb2,
                                           template_lcd2, template_exterior2, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs3, rotate, range_d, threshold, lcd_txt3, exterior_txt3, gylb3,
                                           template_lcd3, template_exterior3, E, result_xml)
            detect_biaowei(1, imgs4, rotate, range_d, threshold, lcd_txt4, exterior_txt4, gylb4, template_lcd4,
                           template_exterior4, E, result_xml)

        elif number_e == '5':
            E, result_xml = detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1,
                                           template_lcd1, template_exterior1, None, None)
            E, result_xml = detect_biaowei(1, imgs2, rotate, range_d, threshold, lcd_txt2, exterior_txt2, gylb2,
                                           template_lcd2, template_exterior2, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs3, rotate, range_d, threshold, lcd_txt3, exterior_txt3, gylb3,
                                           template_lcd3, template_exterior3, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs4, rotate, range_d, threshold, lcd_txt4, exterior_txt4, gylb4,
                                           template_lcd4, template_exterior4, E, result_xml)
            detect_biaowei(1, imgs5, rotate, range_d, threshold, lcd_txt5, exterior_txt5, gylb5, template_lcd5,
                           template_exterior5, E, result_xml)

        elif number_e == '6':

            E, result_xml = detect_biaowei(0, imgs1, rotate, range_d, threshold, lcd_txt1, exterior_txt1, gylb1,
                                           template_lcd1, template_exterior1, None, None)
            E, result_xml = detect_biaowei(1, imgs2, rotate, range_d, threshold, lcd_txt2, exterior_txt2, gylb2,
                                           template_lcd2, template_exterior2, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs3, rotate, range_d, threshold, lcd_txt3, exterior_txt3, gylb3,
                                           template_lcd3, template_exterior3, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs4, rotate, range_d, threshold, lcd_txt4, exterior_txt4, gylb4,
                                           template_lcd4, template_exterior4, E, result_xml)
            E, result_xml = detect_biaowei(1, imgs5, rotate, range_d, threshold, lcd_txt5, exterior_txt5, gylb5,
                                           template_lcd5, template_exterior5, E, result_xml)
            detect_biaowei(1, imgs6, rotate, range_d, threshold, lcd_txt6, exterior_txt6, gylb6, template_lcd6,
                           template_exterior6, E, result_xml)
    except Exception as e:
        E = objectify.ElementMaker(annotate=False)
        result_xml = E.DATA(
            E.METER(
                E.CHECK_RESULT(
                    E.REGION_TYPE("number_e wrong"),
                ),
            ),
        )
        etree.ElementTree(result_xml).write("result.xml", pretty_print=True)
run(img_path, template_path, threshold, rotate, range_d, number_e)
