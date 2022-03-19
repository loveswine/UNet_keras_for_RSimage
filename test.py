from keras.models import load_model
from dataProcess import testGenerator, saveResult, color_dict
import os
import warnings
warnings.filterwarnings('ignore')

#  训练模型保存地址
model_path = r"./model/unet_model.hdf5"
#  测试数据路径
test_iamge_path = "./dataset/test/image"
#  结果保存路径
save_path = "./dataset/test/predict"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (256, 256, 3)
#  生成图像大小
output_size = (256, 256)
#  训练数据标签路径
train_label_path = "./dataset/test/label"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


model=load_model(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_RGB, output_size)