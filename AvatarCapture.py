import cv2
import sys
import os.path
from glob import glob
from matplotlib import pyplot as plt

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48)#识别区大小
                                     )
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        #此处输出裁剪后大小
        face = cv2.resize(face, (256,256))
        #可视化
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
        if opt['opencv-val']==1:
            cv2.namedWindow('AnimeFaceDetect', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("AnimeFaceDetect", image)
            cv2.waitKey(0)
        if opt["PLT-val"]==1:
            cvimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #plt.title("菜鸟教程 - 测试")
            plt.imshow(cvimg)
            plt.show()
        if opt["save-val-image"]==1:
            save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
            cv2.imwrite("out/" + save_filename, image)
        if opt["show-fin-image"]==1:
            cv2.imshow("img",face)
            cv2.waitKey(0)
        if opt["save-fin-image"]==1:
            save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
            cv2.imwrite("Data/" + save_filename, face)

if __name__ == '__main__':
    if os.path.exists('Data') is False:
        os.makedirs('Data')
    if os.path.exists('out') is False:
        os.makedirs('out')
    if os.path.exists('IMG') is False:
        print('file not exists!')
        exit()
    opt = {"opencv-val": 0,
           "PLT-val": 1,
           "show-fin-image":0,
           "save-fin-image":0,
           "save-val-image":0}
    file_list = glob('IMG/*.jpg')
    for filename in file_list:
        try:
            detect(filename)
        except:
            print('A Error happy ...')
        else:
            print(str(filename) + ' done !')
