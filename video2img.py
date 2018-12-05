import time

import cv2

if __name__ == '__main__':
    # 填写视频的绝对路径
    VideoName = "cigar.mp4" 
    vidcap = cv2.VideoCapture(VideoName)
    success, image = vidcap.read()
    start_time = time.time()
    print(start_time)
    flag = 0
    while success:
        end_time = time.time()
        file_name = str(end_time).replace('.', '') + str(flag)
        # 每隔三秒截屏
        if flag%3 == 0 :
            start_time = end_time
            # 保存JGP  的绝对路径
            cv2.imwrite('jpg/' + file_name + ".jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        flag += 1
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break