import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import xml.etree.ElementTree as xmlET
import os
import scipy.misc
from PIL import Image
import numpy as np
import random
from addnoise import add_gasuss_noise,add_haze,ajust_image,ajust_image_hsv,ajust_jpg_quality,add_gasuss_blur
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
# PCA__jiltter
def pca_jiltter(img):
    img = np.asarray(img)
    if(img.ndim == 3):
        img = img/255
        img_size=img.size//3
        img1 = img.reshape(img_size,3)
        img1 = np.transpose(img1)   
        img_conv =  np.cov([img1[0],img1[1],img1[2]])
        lamda,p = np.linalg.eig(img_conv)
        p = np.transpose(p)
        alpha1 = random.normalvariate(0,0.3)
        alpha2 = random.normalvariate(0,0.3)
        alpha3 = random.normalvariate(0,0.3)
        v = np.transpose((alpha1*lamda[0],alpha2*lamda[1],alpha3*lamda[2]))
        
        
        add_num = np.dot(p,v)
        img2 = np.array([img[:,:,0]+add_num[0],img[:,:,1]+add_num[1],img[:,:,2]+add_num[2]])
        img2 = np.swapaxes(img2,0,2)
        img2 = np.swapaxes(img2,0,1)
        return img2
# add gause
def gaussianNoisy(im, mean=0.2, sigma=0.3):

    for _i in range(len(im)):
        im[_i] += random.gauss(mean, sigma)
    return im


def add_gauss(img,mean,sigma):
    if(img.ndim == 3):
       #mean=0.8
       #sigma=0.3
       img = np.asarray(img)
       img.flags.writeable = True 
       width, height = img.shape[:2]
       img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
       img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
       img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
       img[:, :, 0] = img_r.reshape([width, height])
       img[:, :, 1] = img_g.reshape([width, height])
       img[:, :, 2] = img_b.reshape([width, height])
        
    #   Image.fromarray(np.uint8(img)).save(sys.argv[1]+"_addgauss.jpg")
    elif(img.ndim == 2):
        width, height = img.shape[:2]
        img[:, :] = gaussianNoisy(img[:, :].flatten(), mean, sigma)
        img[:, :] = img.reshape([width, height])
    return img

file_path_xml = "xml"
file_path_img = "jpg"
pathDir = os.listdir(file_path_xml)

for idx in range(len(pathDir)): 
    filename = pathDir[idx]
    print(filename)
    tree = xmlET.parse(os.path.join(file_path_xml, filename))
    image_name = os.path.splitext(filename)[0]
    img = cv2.imread(os.path.join(file_path_img, image_name + '.jpg'))  #原始图像

    #im = Image.open(os.path.join(file_path_img, image_name + '.jpg'))
    #print(im)
    #im = im.convert('RGB')
    #im.save(os.path.join(file_path_img, image_name + '.jpg'))

    objs = tree.findall('object')        
    bboxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        
        # Make pixel indexes 0-based
        x1 = int(bbox.find('xmin').text) - 1
        y1 = int(bbox.find('ymin').text) - 1
        x2 = int(bbox.find('xmax').text) - 1
        y2 = int(bbox.find('ymax').text) - 1
        
        label = obj.find('name').text
        bboxes.append([x1,y1,x2,y2]) 
        #print(bboxes)
    

    out = add_haze(img)
    cv2.imwrite(os.path.join(file_path_img,image_name + 'haze.jpg'), out)
    save_xml(image_name + 'haze.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)
    
    out = add_gasuss_noise(img)
    cv2.imwrite(os.path.join(file_path_img,image_name + 'gauss_noise.jpg'), out)
    save_xml(image_name + 'gauss_noise.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)

    out = add_gasuss_blur(img)
    cv2.imwrite(os.path.join(file_path_img,image_name + 'gasuss_blur.jpg'), out)
    save_xml(image_name + 'gasuss_blur.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)
    
    out = ajust_image_hsv(img)
    cv2.imwrite(os.path.join(file_path_img,image_name + 'ajust_hsv.jpg'), out)
    save_xml(image_name + 'ajust_hsv.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)

    out = ajust_image(img)
    cv2.imwrite(os.path.join(file_path_img,image_name + 'ajust.jpg'), out)
    save_xml(image_name + 'ajust.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)
    
    '''
    pcatmp = pca_jiltter(np.array(im))
    pca_jit = Image.fromarray(pcatmp)
    pca_jit.save(os.path.join(file_path_img,image_name + 'pca.jpg'))
    '''
    
    '''
    gautmp = add_gauss(np.array(im),0.5,0.3)
    gauss2 = Image.fromarray(gautmp)
    gauss2.save(os.path.join(file_path_img,image_name + 'gauss05.jpg'))
    print(' gauss05 Img saved')
    #scipy.misc.imsave(os.path.join(file_path_img,image_name + 'gauss.jpg'),pcatmp)
    #cv2.imwrite(os.path.join(file_path_img,image_name + 'pca.jpg'), img)
    save_xml(image_name + 'gauss05.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)

    gautmp = add_gauss(np.array(im),0.2,0.3)
    gauss2 = Image.fromarray(gautmp)
    gauss2.save(os.path.join(file_path_img,image_name + 'gauss02.jpg'))
    print(' gauss02 Img saved')
    #scipy.misc.imsave(os.path.join(file_path_img,image_name + 'gauss.jpg'),pcatmp)
    #cv2.imwrite(os.path.join(file_path_img,image_name + 'pca.jpg'), img)
    save_xml(image_name + 'gauss02.jpg',bboxes,save_dir=file_path_xml,label=label, width=2666, height=2000, channel=3)
    '''








