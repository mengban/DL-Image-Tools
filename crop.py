import cv2
import time
import xml.etree.ElementTree as xmlET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
 
global img
global point1, point2

def save_xml(image_name, bbox, save_dir='path/Annotations',label='steel', width=2666, height=2000, channel=3):
 
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
 
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
 
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
 
    for x, y, x1, y1 in bbox:
        left, top, right, bottom = x, y, x1, y1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
 
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
 
    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)
 
    return

def on_mouse(event, x, y, flags, param):
    global img, point1, point2, img3
    global filename
    global xml_path 
    global img_path 
    global save_path 
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放

         

        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])

        tree = xmlET.parse(os.path.join(xml_path,filename[:].replace('jpg', 'xml')))
        objs = tree.findall('object')  
        bboxes = []
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            
            # Make pixel indexes 0-based 减去偏移量
            x1 = int(int(bbox.find('xmin').text)/2 - 1 - min_x)*2
            y1 = int(int(bbox.find('ymin').text)/2 - 1 - min_y)*2
            x2 = int(int(bbox.find('xmax').text)/2 - 1 - min_x)*2
            y2 = int(int(bbox.find('ymax').text)/2 - 1 - min_y)*2
            label = obj.find('name').text
            bboxes.append([x1,y1,x2,y2]) 
        save_xml(os.path.join(save_path,'crop_'+filename),bboxes,save_dir='.',label=label, width=2666, height=2000, channel=3)


        cut_img = img3[min_y*2:(min_y+height)*2, min_x*2:(min_x+width)*2]
        cv2.imwrite(os.path.join(save_path,'crop_'+filename), cut_img)



def main():
    global img
    global img3
    global filename
    global xml_path 
    global img_path 
    global save_path 
    xml_path = 'xml'
    img_path = 'jpg'
    save_path = 'save'

    #img = cv2.imread('demo.jpg')
    #img3 = img.copy()
    #size = img.shape
    #img = cv2.resize(img,(size[1]//2,size[0]//2))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    #cv2.imshow('image', img)
    while True:
        filelist = os.listdir(img_path)
        for _ in filelist:
            print(_)
            print(time.time())
            filename = _
            img = cv2.imread(os.path.join(img_path,filename))
            img3 = img.copy()
            size = img.shape
            img = cv2.resize(img,(size[1]//2,size[0]//2))
            cv2.imshow('image', img)
            cv2.waitKey(3000)
 
if __name__ == '__main__':
    main()
    