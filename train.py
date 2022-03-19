import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet,attunet

from dataProcess import trainGenerator, color_dict
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,Callback
import matplotlib.pyplot as plt
import datetime,time
import xlwt
import os
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')

tf.logging.set_verbosity(tf.logging.ERROR)
'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "./dataset/train/image"
#  训练数据标签路径
train_label_path = "./dataset/train/label"
#  验证数据图像路径
validation_image_path = "./dataset/val/image"
#  验证数据标签路径
validation_label_path = "./dataset/val/label"

'''
模型相关参数
'''
#  批大小
batch_size = 4
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (256, 256, 3)
#  训练模型的迭代总轮数
epochs = 10
#  初始学习率
learning_rate = 1e-4
#  预训练模型地址
premodel_path = None
#  训练模型保存地址
model_path = "./model/unet_model.hdf5"

#  训练数据数目
train_num = len(os.listdir(train_image_path))
#  验证数据数目
validation_num = len(os.listdir(validation_image_path))
#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path,
                                 train_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  定义模型
model = unet(pretrained_weights=premodel_path,
             input_size=input_size,
             classNum=classNum,
             learning_rate=learning_rate)
# model = seg_hrnet(pretrained_weights = premodel_path,
#                  input_size = input_size, 
#                  classNum = classNum, 
#                  learning_rate = learning_rate)
#  打印模型结构
# model.summary()
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
#  当5个epoch过去而val_loss不下降时，学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.time=[]
        self.traintime_start=time.time()
    def on_train_end(self, logs=None):
        self.traintime=time.time()-self.traintime_start
    def on_epoch_begin(self, epoch, logs=None):
        self.epochtime_start=time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.time.append(time.time()-self.epochtime_start)
        
time_callback = TimeHistory()
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor='loss',
                                   verbose=2,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only=True)


#  模型训练
history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              callbacks=[early_stopping, model_checkpoint, reduce_lr,time_callback],
                              validation_data=validation_data,
                              validation_steps=validation_steps,
                              verbose=2
                              )


log_time = "训练总时间: " + str(time_callback.traintime/ 60) + "m"
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
print(log_time)
with open('TrainTime_%s.txt' % time, 'w') as f:
    f.write(log_time)

#  保存并绘制loss,acc
history.history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(acc)):
    sheet.write(i, 0, str(acc[i]))
    sheet.write(i, 1, str(val_acc[i]))
    sheet.write(i, 2, str(loss[i]))
    sheet.write(i, 3, str(val_loss[i]))
book.save(r'AccAndLoss_%s.xls' % time)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy_%s.png" % time, dpi=300)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss_%s.png" % time, dpi=300)
plt.show()

plt.figure()
import numpy as np
plt.plot(epochs, np.array(time_callback.time), 'r', label='Training time')

plt.title('Training time for each epoch')
plt.legend()
plt.show()


