# coding = utf-8
import numpy as np 
import cv2
def add_gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out
def add_haze(image, t=0.6, A=1):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    out = image*t + A*255*(1-t)
    return out
def ajust_image(image, cont=1, bright=0):
    '''
        调整对比度与亮度
        cont : 对比度，调节对比度应该与亮度同时调节
        bright : 亮度
    '''
    out = np.uint8(np.clip((cont * image + bright), 0, 255))
    # tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
    return out
def ajust_image_hsv(image, h=1, s=1, v=0.8):
    '''
        调整HSV通道，调整V通道以调整亮度
        各通道系数
    '''
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H2 = np.uint8(H * h)
    S2 = np.uint8(S * s)
    V2 = np.uint8(V * v)
    hsv_image = cv2.merge([H2, S2, V2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return out
def ajust_jpg_quality(image, q=100, save_path=None):
    '''
        调整图像JPG压缩失真程度
        q : 压缩质量 0~100
    '''
    if save_path is None:
        cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out = cv2.imread('jpg_tmp.jpg')
        return out
    else:
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
def add_gasuss_blur(image, kernel_size=(3, 3), sigma=0.1):
    '''
        添加高斯模糊
        kernel_size : 模糊核大小
        sigma : 标准差
    '''
    out = cv2.GaussianBlur(image, kernel_size, sigma)
    return out
def test_methods():
    img = cv2.imread('test.jpg')
    out = add_haze(img)
    cv2.imwrite("add_haze.jpg", out)
    out = add_gasuss_noise(img)
    cv2.imwrite("add_gasuss_noise.jpg", out)
    out = add_gasuss_blur(img)
    cv2.imwrite("add_gasuss_blur.jpg", out)
    out = ajust_image(img)
    cv2.imwrite("ajust_image.jpg", out)
    out = ajust_image_hsv(img)
    cv2.imwrite("ajust_image_hsv.jpg", out)
    ajust_jpg_quality(img, save_path='ajust_jpg_quality.jpg')

test_methods()