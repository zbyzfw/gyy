import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = "stdlib"
import sys
import cv2,json
import time
import glob
import warnings
import configparser
from lxml import etree, objectify
from lxml.etree import Element
import logging
import datetime
from tools.infer.predict import detect_roi


warnings.filterwarnings("ignore")
if not os.path.exists('barLog'):
    os.makedirs('barLog')
logfilename = f"barLog/Log{datetime.date.today().strftime('%Y%m%d')}.log"
logging.basicConfig(level=logging.INFO, filename=logfilename, format="%(asctime)s: %(module)s: %(message)s")
cf = configparser.ConfigParser()


# img_path = sys.argv[1]
# ini_path = sys.argv[2]
# rotate = sys.argv[3]
# type_name = sys.argv[4]
# extra = sys.argv[5]
img_path = '333image/'
ini_path = '333image/config8.ini'
rotate = '3'
type_name = '1'
# extra = '0'

fp = open('meter.json', encoding='utf-8')
meter_data = json.load(fp)
meter1 = meter_data["20版三相"]
meter2 = meter_data["20版单相"]
meter3 = meter_data["13版三相"]
meter4 = meter_data["13版单相"]


# 实例化一个xml节点
E = objectify.ElementMaker(annotate=False)

def detect_lcd(path,child1):
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
    top_x = int(cf.get(f"Location1", "topX"))
    top_y = int(cf.get(f"Location1", "topY"))
    width = int(cf.get(f"Location1", "width"))
    height = int(cf.get(f"Location1", "height"))
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
    # s_meter = ''.join([s[0] for s in results])
    # print(s_meter)
    if results:
        # 所有检测到的文字
        results_str = ''.join([res[0] for res in results])
        print(results_str)
        if type_name == '1':
            for char in set(meter1[0]):
                if results_str.count(char) < meter1[0].count(char):
                    if char not in meter1[1]:
                        set_meter.append(char)
        elif type_name == '2':
            for char in set(meter2[0]):
                if results_str.count(char) < meter2[0].count(char):
                    if char not in meter2[1]:
                        set_meter.append(char)
        elif type_name == '3':
            for char in set(meter3[0]):
                if results_str.count(char) < meter3[0].count(char):
                    if char not in meter3[1]:
                        set_meter.append(char)
        elif type_name == '4':
            for char in set(meter4[0]):
                if results_str.count(char) < meter4[0].count(char):
                    if char not in meter4[1]:
                        set_meter.append(char)
        else:
            tree41 = E.BAR_CODE(f"{path} ini error")
            tree42 = E.CHECK_RESULT(
                E.REGION_TYPE("1"),
                E.CHECK_FLAG("NG, 请输入要识别的电表类型")
            )
            child1.append(tree41)
            child1.append(tree42)
            return
        print(set_meter)
        if set_meter:
            tree41 = E.BAR_CODE(f"{path}")
            tree42 = E.CHECK_RESULT(
                E.REGION_TYPE("1"),
                E.CHECK_FLAG(f"NG,{set_meter}")
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
            # cv2.imshow("3", roi)
            # cv2.waitKey(1000)
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
    elif not os.path.exists('./meter2.json'):
        # print('配置文件不存在')
        tree41 = E.BAR_CODE(f"meter.json")
        tree42 = E.CHECK_RESULT(
            E.REGION_TYPE("1"),
            E.CHECK_FLAG(f"NG, json文件不存在，请联系管理员")
        )
        child1.append(tree41)
        child1.append(tree42)
    else:
        imgs = glob.glob(img_path+'/*.jpg')  # 所有目录
        # 创建xml文件根节点
        cf.read(ini_path)
        for img in imgs:
            detect_lcd(img, child1)
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