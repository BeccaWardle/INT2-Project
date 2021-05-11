#encoding:utf-8
#CIFAR-10数据集可视化保存到本地
#https://blog.csdn.net/qq_41895190/article/details/103522142
import cv2 as cv
import numpy as np
from cv2 import cv2

# 解压缩，返回解压后的字典
def unpickle(file):
    import _pickle as cpickle 
    fo = open(file, 'rb')
    dict = cpickle.load(fo, encoding='bytes')
    fo.close()
    return dict

# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
for j in range(2, 3):
    dataName = "/Users/jg/Documents/GitHub/INT2-Project/data/cifar-10-batches-py/data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")

    for i in range(2, 3):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1,2,0)    # 读取image
        picName = '/Users/jg/Documents/GitHub/INT2-Project/documents/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        cv2.imwrite(picName, img)
    print(dataName + " loaded.")

print("test_batch is loading...")

