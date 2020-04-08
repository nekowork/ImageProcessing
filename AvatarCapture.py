import cv2
import sys
import os.path
from glob import glob

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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("AnimeFaceDetect", image)
        cv2.waitKey(0)
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("out/" + save_filename, image)
        #显示裁剪完成图片
        #cv2.imshow("img",face)
        #cv2.waitKey(0)
        #保存裁剪图片打开此注释
        #save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("Data/" + save_filename, face)

if __name__ == '__main__':
    if os.path.exists('Data') is False:
        os.makedirs('Data')
    if os.path.exists('out') is False:
        os.makedirs('out')
    if os.path.exists('IMG') is False:
        print('file not exists!')
        exit()
    file_list = glob('IMG/*.jpg')
    for filename in file_list:
        try:
            detect(filename)
        except:
            print('A Error happy ...')
        else:
            print(str(filename)+' done !')
