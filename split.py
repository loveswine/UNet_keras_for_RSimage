# -*- coding: utf-8 -*-
"""
@author: xqxqxxq
"""
import os
import gdal
import numpy as np


#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


'''
滑动窗口裁剪函数
TifPath 影像路径
SavePath 裁剪后保存目录
CropSize 裁剪尺寸
RepetitionRate 重复率
'''


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 1
    #  裁剪图片,重复率为RepetitionRate
    
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #  写图像
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
            #  文件名 + 1
            new_name = new_name + 1
    #  向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        #  写图像
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1
    #  向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 2):
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        #  文件名 + 1
        new_name = new_name + 1
    #  裁剪右下角
    if (len(img.shape) == 2):
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    new_name = new_name + 1


def RandomCrop(ImagePath, LabelPath, IamgeSavePath, LabelSavePath, CropSize, CutNum):
    dataset_img = readTif(ImagePath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取哟昂数据
    dataset_label = readTif(LabelPath)
    label = dataset_label.ReadAsArray(0, 0, width, height)  # 获取标签数据
    
    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    fileNum = len(os.listdir(IamgeSavePath))
    new_name = fileNum + 1
    import random
    while (new_name < CutNum + fileNum + 1):
        #  生成剪切图像的左上角XY坐标
        UpperLeftX = random.randint(0, height - CropSize)
        UpperLeftY = random.randint(0, width - CropSize)
        if (len(img.shape) == 2):
            imgCrop = img[UpperLeftX: UpperLeftX + CropSize,
                      UpperLeftY: UpperLeftY + CropSize]
        else:
            imgCrop = img[:,
                      UpperLeftX: UpperLeftX + CropSize,
                      UpperLeftY: UpperLeftY + CropSize]
        if (len(label.shape) == 2):
            labelCrop = label[UpperLeftX: UpperLeftX + CropSize,
                        UpperLeftY: UpperLeftY + CropSize]
        else:
            labelCrop = label[:,
                        UpperLeftX: UpperLeftX + CropSize,
                        UpperLeftY: UpperLeftY + CropSize]
        writeTiff(imgCrop, geotrans, proj, IamgeSavePath + "/%d.tif" % new_name)
        writeTiff(labelCrop, geotrans, proj, LabelSavePath + "/%d.tif" % new_name)
        new_name = new_name + 1

#  将影像1裁剪为重复率为0.1的256×256的数据集
datapath=r'your image path'
savepath='./dataset/test'
picsize=256
files=os.listdir(datapath+'/image')

for file in files:
    TifCrop(os.path.join(datapath,'image',file),
            os.path.join(savepath,'image'), picsize, 0.1)
    TifCrop(os.path.join(datapath,'label',file.replace('image','labels')),
            os.path.join(savepath,'label'), picsize, 0.1)

print('done')

'''
去除空图像
'''
# labelimgs=os.listdir(savepath+'/label')
#
# for file in labelimgs:
#     dataset_img = readTif(os.path.join(savepath,'label',file))
#     img = dataset_img.ReadAsArray(0, 0, picsize, picsize)
#     if np.sum(img)/(picsize**2*3*255)>0.8:
#         del dataset_img
#         os.remove(os.path.join(savepath,'label',file))
#         os.remove(os.path.join(savepath, 'image', file))
    

    