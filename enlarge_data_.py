#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import glob
import os
import PIL
from PIL import Image
from sklearn import cross_validation
from skimage import exposure
import random
import scipy.misc
import scipy.ndimage as ndi
import cv2

height =100
weight = 100

cur_path = os.getcwd()
train_norm_path = ''   #原始图片路径
train_ad_path = ''     


train_path = ''        #增强数据路径

test_norm_path = ''
test_ad_path = ''


test_path = ''

cigar_input = 'data_input/'
cigar_output = 'data_output/'

EXTS = 'jpg', 'jpeg', 'gif', 'png', 'BMP', 'PNG', 'bmp'

##inverse

def inverse(img):
    if(img.ndim == 3):

        b = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
        g = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
        r = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)

        b[:,:] = np.full((img.shape[0],img.shape[1]), 255) - img[:,:,0]  #  b   
        g[:,:] = np.full((img.shape[0],img.shape[1]), 255) - img[:,:,1]  #  g   
        r[:,:] = np.full((img.shape[0],img.shape[1]), 255) - img[:,:,2]  #  r   



        mergedByNp = np.dstack([b,g,r])
        merged = cv2.merge([b,g,r])
    elif(img.ndim == 2):
        b = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)

        merged[:,:] = np.full((img.shape[0],img.shape[1]), 255) - img[:,:]  #  b   
    return merged

##zoom
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,transform_matrix,channel_axis=0,fill_mode='nearest',cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


##pca_jiltter
def pca_jiltter(img):
    img = np.asarray(img,dtype="float32")
    if(img.ndim == 3):
        img = img/255
        img_size=img.size/3
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

##add_gauss
 

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
    #    Image.fromarray(im).save(sys.argv[1]+"_addgauss.jpg")

def load_image(image_file_path, save_path, label,aug):
    image_list = []
    os.chdir(image_file_path)
    for ext in EXTS:
        image_list.extend(glob.glob('*.%s' % ext))
    data = []
    data_y = []
    data_name = []
    for image_name in image_list:
        try:
            #print(os.path.abspath('.'))
            print(image_name)
            im = Image.open(image_name)
            #print(im)
            im = im.convert('RGB')
            im.save(save_path+image_name)
            data_y.append(label)
            data_name.append(save_path+image_name)
            #pic = im.convert('L')
            #pic = pic.resize((height,weight),PIL.Image.ANTIALIAS)
            #pic = np.array(pic).reshape(1,height*weight)

            if aug ==1:

                #flip = np.fliplr(pic)
                file_prefiex = save_path + image_name.split('.')[0] + '_'
                #print('prefiex:',file_prefiex)
                im_rotate = im.rotate(45)
                im_rotate.save(file_prefiex+"ro45.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"ro45.jpg")

                im_lr = im.transpose(Image.FLIP_LEFT_RIGHT)#leftright
                im_lr.save(file_prefiex+"lr.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"lr.jpg")

                im_ud = im.transpose(Image.FLIP_TOP_BOTTOM)#updown
                im_ud.save(file_prefiex+"ud.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"ud.jpg")

                im_90 = im.transpose(Image.ROTATE_90)#90
                im_90.save(file_prefiex+"ro90.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"ro90.jpg")

                im_180 = im.transpose(Image.ROTATE_180)#180
                im_180.save(file_prefiex+"ro180.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"ro180.jpg")

                im_270 = im.transpose(Image.ROTATE_270)#270
                im_270.save(file_prefiex+"ro720.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"ro720.jpg")

                gam1 = Image.fromarray(exposure.adjust_gamma(np.array(im)))
                gam1.save(file_prefiex+"gam1.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"gam1.jpg")

                
                gam2 = Image.fromarray(exposure.adjust_gamma(np.array(im)))
                gam2.save(file_prefiex+"gam2.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"gam2.jpg")
                

                log = Image.fromarray(exposure.adjust_log(np.array(im)))
                log.save(file_prefiex+"log.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"log.jpg")

                mat1 = Image.fromarray(exposure.rescale_intensity(np.array(im)))
                mat1.save(file_prefiex+"mat1.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"mat1.jpg")

                shift_d = Image.fromarray(np.roll(np.array(im),20,axis=0))
                shift_d.save(file_prefiex+"shift_d.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"shift_d.jpg")

                shift_u = Image.fromarray(np.roll(np.array(im),-20,axis=0))
                shift_u.save(file_prefiex+"shift_u.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"shift_u.jpg")


                shift_l = Image.fromarray(np.roll(np.array(im),-20,axis=1))
                shift_l.save(file_prefiex+"shift_l.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"shift_l.jpg")

                shift_r = Image.fromarray(np.roll(np.array(im),20,axis=1))
                shift_r.save(file_prefiex+"shift_r.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"shift_r.jpg")

                #zoom_small
                zoomtmp = random_zoom(np.array(im),(0.3,0.3))
                zoom3 = Image.fromarray( zoomtmp )
                zoom3.save(file_prefiex+"zoom0.3.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"zoom0.3.jpg")


                ##zoom_small
                zoomtmp = random_zoom(np.array(im),(0.5,0.5))
                zoom5 = Image.fromarray( zoomtmp )
                zoom5.save(file_prefiex+"zoom0.5.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"zoom0.5.jpg")

                ##zoom_big
                
                zoomtmp = random_zoom(np.array(im),(2,2))
                zoom2 = Image.fromarray(zoomtmp)
                zoom2.save(file_prefiex+"zoom2.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"zoom2.jpg")
                

                ##pca
                
                pcatmp = pca_jiltter(np.array(im))
                pca_jit = Image.fromarray( pcatmp )
                scipy.misc.imsave(file_prefiex+"pca.jpg",pcatmp)
                pca_jit.save(file_prefiex+"pca.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"pca.jpg")
                

                #add_gauss_mean0.2
                gautmp = add_gauss(np.array(im),0.2,0.3)
                gauss2 = Image.fromarray(gautmp )
                gauss2.save(file_prefiex+"gauss0.2.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"gauss0.2.jpg")


                #add_gauss_mean0.5
                
                gautmp = add_gauss(np.array(im),0.5,0.3)
                gauss5 = Image.fromarray(gautmp )
                gauss5.save(file_prefiex+"gauss0.5.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"gauss0.5.jpg")


                ##inverse
                invtmp = inverse(np.array(im) )
                inversepic = Image.fromarray(invtmp )
                inversepic.save(file_prefiex+"inverse.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"inverse.jpg")
                '''



                ##inverse270
                '''
                pic = im.convert('L')
                pic = pic.resize((height,weight),PIL.Image.ANTIALIAS)
                pic = np.array(pic).reshape(1,height*weight)
                im_flip = np.fliplr(pic)
                flip = Image.fromarray(im_flip.reshape(height,weight))
                flip.save(file_prefiex+"flip.jpg")
                data_y.append(label)
                data_name.append(file_prefiex+"flip.jpg")
                


            else:
                data.append(pic[0].tolist())
                data_y.append(label)
                data_name.append(image_name)
                pass
        except Exception as e:
            print ("load_image Exception:",e)

    data_matrix = np.array(data)
    data_y = np.array(data_y).reshape(len(data_y),1)
    data_name = np.array(data_name)
    return  (data_y,data_name)


def main():
	##train data
   
    load_image(data_input,data_output,1,1)

if __name__== "__main__":
    main()
