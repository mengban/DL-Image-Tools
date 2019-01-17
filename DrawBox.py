''''
功能:实现模型结果的可视化
用法：将标注文件、图片分别放至文件夹xml、jpg中 可视文件会保存至save
'''
import os
import xml.etree.ElementTree as xmlET
import cv2

file_path_img = 'data/jpg'
file_path_xml = 'data/xml'
save_file_path = 'data/save'

def drawBox():
    pathDir = os.listdir(file_path_xml)
    for idx in range(len(pathDir)): 
        filename = pathDir[idx]
        tree = xmlET.parse(os.path.join(file_path_xml, filename))
        image_name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(file_path_img, image_name + '.jpg')) 
        objs = tree.findall('object')        
        
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
    
            label = obj.find('name').text 
    
            # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
        
            # 标注文本
            font = cv2.FONT_HERSHEY_COMPLEX
            # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
            cv2.putText(img, label, (x1, y1), font, 1, (0,0,255), 2)
        
        cv2.imwrite(os.path.join(save_file_path, image_name + '.jpg'), img)
if __name__=='__main__':
    drawBox()