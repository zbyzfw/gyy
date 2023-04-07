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
from tools.infer.predict2 import detect_roi2
from flask import Flask

app = Flask(__name__)


@app.route('/detect',methods=['GET'])
def detect():
    request.args.get('key')
    return {'message': 'hello world'}


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
img_path = './photo/60'
ini_path = './photo/ini60'
rotate = '0'
type_name = '1'
range_d='12'
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
        img = img.copy()
    w1= img.shape[0]
    h1=img.shape[1]
    print(w1,h1)
    if range_d.find('1')!=-1:
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
        set_meter = []
        results = detect_roi(roi)
        # print(results)
        if results:
            # 识别结果转换为列表
            results_news = [res[0] for res in results]
            # for res in results:
            if type_name == '1':
                # 替换识别错误的字符
                # print(meter1[-1].split(','))
                for meter in meter1[-1].split(','):
                    if meter:
                        temp = meter.split('>')
                        results_news = [re.sub(temp[0],temp[1],res) for res in results_news]
                # print(results_news)
                for res in meter1[:-1]:
                    if res not in results_news:
                        # print(res)
                        set_meter.append(res)
            elif type_name == '2':
                for meter in meter2[-1].split(','):
                    if meter:
                        temp = meter.split('>')
                        results_news = [res.replace(temp[0], temp[1]) for res in results_news]

                for res in meter2[:-1]:
                    if res not in results_news:
                        set_meter.append(res)
            elif type_name == '3':
                for meter in meter3[-1].split(','):
                    if meter:
                        temp = meter.split('>')
                        results_news = [res.replace(temp[0], temp[1]) for res in results_news]

                for res in meter3[:-1]:
                    if res not in results_news:
                        set_meter.append(res)
            elif type_name == '4':
                for meter in meter4[-1].split(','):
                    if meter:
                        temp = meter.split('>')
                        results_news = [res.replace(temp[0], temp[1]) for res in results_news]

                for res in meter4[:-1]:
                    if res not in results_news:
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
            cv2.waitKey(1000)
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
    if range_d.find('2') != -1:
        if type_name == '2' or type_name == '4':
            if rotate == '3':
                roi1 = img[0:w1, top_x + width-10:h1]
                roi2 = img[0:w1, 0:top_x-100]
            if rotate == '0':
                roi1 = img[0:top_y, 0:h1]
                roi2 = img[top_y+height+100:w1, 0:h1]
            if rotate == '1':
                roi1 = img[0:w1, 0:top_x+10]
                roi2 = img[0:w1, top_x+width+100:h1]
            if rotate == '2':
                roi1 = img[top_y+height-10:w1, 0:h1]
                roi2 = img[0:top_y-100, 0:h1]

        else:
            if rotate == '3':
                roi1 = img[top_y + height-10:w1, top_x:top_x + width]
                roi2 = img[0:w1, 0:top_x-100]
            if rotate == '0':
                roi1 = img[top_y:top_y + height, top_x+width:h1]
                roi2 = img[top_y + height+100:w1, 0:h1]
            if rotate == '1':
                roi1 = img[0:top_y-10, top_x:top_x + width]
                roi2 = img[0:w1, top_x + width+100 : h1]
            if rotate == '2':
                roi1 = img[top_y:w1, 0:top_x+10]
                roi2 = img[0:top_y-100, 0:h1]
        # top_x1 = int(config.get(f"Location2", "topX"))
        # top_y1 = int(config.get(f"Location2", "topY"))
        # width1 = int(config.get(f"Location2", "width"))
        # height1 = int(config.get(f"Location2", "height"))
        # try:
        #     roi2 = img[top_y1:top_y1 + height1, top_x1:top_x1 + width1, ]
        # except Exception as e:
        #     print('不存在')
        if rotate == '1':
            roi1 = cv2.flip(cv2.transpose(roi1), 1)
        elif rotate == '2':
            roi1 = cv2.flip(roi1, -1)
        elif rotate == '3':
            roi1= cv2.flip(cv2.transpose(cv2.flip(roi1, -1)), 1)
        if rotate == '1':
            roi2 = cv2.flip(cv2.transpose(roi2), 1)
        elif rotate == '2':
            roi2 = cv2.flip(roi2, -1)
        elif rotate == '3':
            roi2 = cv2.flip(cv2.transpose(cv2.flip(roi2, -1)), 1)
        results1 = detect_roi2(roi1)
        results2 = detect_roi2(roi2)
        print(results1)
        print(results2)
        cv2.imshow("1", roi1)
        cv2.waitKey(1000)
        cv2.imshow("2", roi2)
        cv2.waitKey(1000)


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
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
