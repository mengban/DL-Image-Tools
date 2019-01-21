''''
功能:实现模型结果的可视化
用法：将标注文件、图片分别放至文件夹xml、jpg中 可视文件会保存至save
'''
import os
import xml.etree.ElementTree as xmlET
import cv2

file_path_img = 'jpg'
file_path_xml = 'xml'
save_file_path = 'data/save'

def voc2txt():
    with open('train_only.csv','w') as f:
        pathDir = os.listdir(file_path_xml)
        for idx in range(len(pathDir)): 
            filename = pathDir[idx]
            print(filename)
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
                f.write(image_name + '.jpg' + ',' + str(x1) +','+ str(y1) + ',' 
                +  str(x2) + ',' + str(y2) + ',' + "gangjin"+"\n")
if __name__=='__main__':
    voc2txt()