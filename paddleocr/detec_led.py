import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = "stdlib"
import cv2
import sys, json
import time, re
import glob
import warnings
import configparser
import numpy as np
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
img_path = 'photo/2001'
ini_path = 'photo/ini2001'
rotate = '3'
type_name = '12'
number_e = '2'
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
    # cv2.imwrite('D:/378/', roi)
    set_meter = []
    # roi1=cv2.resize(roi,(860,514))
    # if type_name.find('1') != -1:

    results = detect_roi(roi)
    # print(results)
    if results:
        # 识别结果转换为列表
        results_news = [res[0] for res in results]
        # for res in results:
        if type_name.find('1') != -1:
            # 替换识别错误的字符
            for meter in meter1[-1].split(','):
                if meter:
                    temp = meter.split('>')
                    results_news = [re.sub(temp[0],temp[1],res) for res in results_news]
            results_no_points = [res.replace(".", "") for res in results_news]
            # print(results_news)
            for i, res in enumerate(meter1[:-1]):
                # 如果是json文件中第一个元素,对比原检测结果,如果不是,对比去除点的检测结果
                if i == 0:
                    if res not in results_news:
                        # print('不存在的8',res)
                        set_meter.append(res)
                else:
                    if res not in results_no_points:
                        set_meter.append(res)
        elif type_name == '2':
            for meter in meter2[-1].split(','):
                if meter:
                    temp = meter.split('>')
                    results_news = [re.sub(temp[0], temp[1],res) for res in results_news]
            results_no_points = [res.replace(".", "") for res in results_news]

            for i, res in enumerate(meter2[:-1]):
                # 如果是json文件中第一个元素,对比原检测结果,如果不是,对比去除点的检测结果
                if i == 0:
                    if res not in results_news:
                        set_meter.append(res)
                else:
                    if res not in results_no_points:
                        set_meter.append(res)
        elif type_name == '3':
            for meter in meter3[-1].split(','):
                if meter:
                    temp = meter.split('>')
                    results_news = [re.sub(temp[0], temp[1],res) for res in results_news]
            results_no_points = [res.replace(".", "") for res in results_news]

            for i, res in enumerate(meter3[:-1]):
                # 如果是json文件中第一个元素,对比原检测结果,如果不是,对比去除点的检测结果
                if i == 0:
                    if res not in results_news:
                        set_meter.append(res)
                else:
                    if res not in results_no_points:
                        set_meter.append(res)
        elif type_name == '4':
            for meter in meter4[-1].split(','):
                if meter:
                    temp = meter.split('>')
                    results_news = [re.sub(temp[0], temp[1],res) for res in results_news]
            results_no_points = [res.replace(".", "") for res in results_news]

            for i, res in enumerate(meter4[:-1]):
                # 如果是json文件中第一个元素,对比原检测结果,如果不是,对比去除点的检测结果
                if i == 0:
                    if res not in results_news:
                        set_meter.append(res)
                else:
                    if res not in results_no_points:
                        set_meter.append(res)
        else:
            tree41 = E.BAR_CODE(f"{path} ini error")
            tree42 = E.CHECK_RESULT(
                E.REGION_TYPE("1"),
                E.CHECK_FLAG("NG, 请输入要识别的电表类型")
            )
            child1.append(tree41)
            child1.append(tree42)
            return
        # print(set_meter)
        if set_meter:
            tree41 = E.BAR_CODE(f"{path}")
            tree42 = E.CHECK_RESULT(
                E.REGION_TYPE("1"),
                E.CHECK_FLAG(f"NG,缺字{set_meter}")
            )
            child1.append(tree41)
            child1.append(tree42)
        else:
            tree41 = E.BAR_CODE(f"{path}")
            tree42 = E.CHECK_RESULT(
                E.REGION_TYPE("1"),
                E.CHECK_FLAG(f"OK")
            )
            child1.append(tree41)
            child1.append(tree42)
        cv2.imshow("3", roi)
        cv2.waitKey(100)
        # print(set_meter)

        logging.info(f'接收参数：[图片路径：{path}，配置文件路径：{ini_path} ]\n识别结果：{results}\n')
    else:
        tree41 = E.BAR_CODE(f"{path}")
        tree42 = E.CHECK_RESULT(
            E.REGION_TYPE("1"),
            E.CHECK_FLAG(f"NG,BLACK")
        )
        child1.append(tree41)
        child1.append(tree42)
    logging.info(f'接收参数：[图片路径：{path}，配置文件路径：{ini_path} ]\n识别结果：{results}\n')
    if type_name.find('2') != -1:
        # if judge_type == 1:
        #     light = gray[int(bottom_right2[1]) + int((bottom_right2[1] - top_left2[1]) / 7):int(
        #         bottom_right2[1] + int((bottom_right2[1] - top_left2[1]) / 2)),
        #             int(top_left2[0]) + int((bottom_right2[0] - top_left2[0]) / 4):int(top_left2[0]) + int(
        #                 (bottom_right2[0] - top_left2[0]) / 2)]
        # else:
        # light = gray[int(bottom_right2[1]):int(bottom_right2[1] + int((bottom_right2[1] - top_left2[1]) * 2 / 5)),
        #         int(top_left2[0]) + int((bottom_right2[0] - top_left2[0]) * 5 / 12):int(top_left2[0]) + int(
        #             (bottom_right2[0] - top_left2[0]) * 7 / 12)]
        # light=gray[int(top_x+height*5/12):int(top_x+height*7/12),int(top_y+width):int(top_y+width+width*2/5)]
        if rotate == '1':
            light = gray[int(top_y + height * 5 / 12):int(top_y + height * 7 / 12),top_x + width:int(top_x + width + width * 2 / 6)]
        if rotate == '3':
            light = gray[int(top_y + height * 5 / 12):int(top_y + height * 7 / 12),int(top_x-(width * 2 / 6)):top_x]
        h5, w5 = light.shape[:2]
        # print("维度",light.shape)
        # 霍夫变换找圆
        if len(light.shape) == 3:
            light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
        # light = cv2.adaptiveThreshold(light, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -5)
        cv2.imshow("light", light)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
            # print(th_img0==0)
            # print(th > (w6 * h6 / 2))
            if th > (w6 * h6 / 2):
                flag3 = 0
            else:
                flag3 = 1
        else:
            th = np.sum(light < 70)
            # print(th , (w5 * h5 / 2))
            if th > (w5 * h5 / 2):
                flag3 = 0
            else:
                flag3 = 1
        # tree41 = E.BAR_CODE(f"{path}")
        tree43 = E.CHECK_RESULT(
            E.REGION_TYPE("2"),
            # E.CHECK_FLAG("OK")
            E.CHECK_FLAG("OK" if flag3==1 else "NG")
        )
        # child1.append(tree41)
        child1.append(tree43)
        # print(flag3)

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
                detect_lcd(img, cf, child1)
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
