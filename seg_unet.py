from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D,  Dropout, concatenate, merge, UpSampling2D,Dense
from keras.optimizers import Adam
from keras.layers import Activation,Reshape,Add,Concatenate,Lambda,multiply
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,MaxPooling2D
import keras.backend as K

def activation(x, func='relu'):
    '''
    Activation layer.
    '''
    return Activation(func)(x)

def CBAM(x, ratio=4, attention='both'): # 8
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    if attention == 'channel':
        cbam_feature = CBAM_channel_attention(x, ratio)
    elif attention == 'spatial':
        cbam_feature = CBAM_spatial_attention(x)
    else:
        cbam_feature = CBAM_channel_attention(x, ratio)
        cbam_feature = CBAM_spatial_attention(cbam_feature)

    return cbam_feature

def CBAM_channel_attention(x, ratio=8):
    channel = x._keras_shape[-1]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='glorot_uniform', #he_normal
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='glorot_uniform',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = activation(cbam_feature, 'sigmoid')

    return multiply([x, cbam_feature])


def CBAM_spatial_attention(x):
    kernel_size = 7

    channel = x._keras_shape[-1]
    cbam_feature = x

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)

    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([x, cbam_feature])


def attunet(pretrained_weights=None, input_size=(256, 256, 4), classNum=2, learning_rate=1e-5):
    inputs = Input(input_size)
    #  2D卷积层
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1))
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2))
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3))
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))
    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4))
    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    
    try:
        merge6 = concatenate([CBAM(drop4), up6], axis=3)
    except:
        merge6 = merge([CBAM(drop4), up6], mode='concat', concat_axis=3)
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    try:
        merge7 = concatenate([CBAM(conv3), up7], axis=3)
    except:
        merge7 = merge([CBAM(conv3), up7], mode='concat', concat_axis=3)
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))
    
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    try:
        merge8 = concatenate([CBAM(conv2), up8], axis=3)
    except:
        merge8 = merge([CBAM(conv2), up8], mode='concat', concat_axis=3)
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    try:
        merge9 = concatenate([CBAM(conv1), up9], axis=3)
    except:
        merge9 = merge([CBAM(conv1), up9], mode='concat', concat_axis=3)
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    
    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    #  如果有预训练的权重
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model



def unet(pretrained_weights = None, input_size = (256, 256, 4), classNum = 2, learning_rate = 1e-5):
    inputs = Input(input_size)
    #  2D卷积层
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs))
    conv1 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1))
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1))
    conv2 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2))
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2))
    conv3 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3))
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3))
    conv4 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4))
    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4))
    conv5 = BatchNormalization()(Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    try:
        merge6 = concatenate([drop4,up6],axis = 3)
    except:
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6))
    conv6 = BatchNormalization()(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6))
 
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    try:
        merge7 = concatenate([conv3,up7],axis = 3)
    except:
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7))
    conv7 = BatchNormalization()(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7))
 
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    try:
        merge8 = concatenate([conv2,up8],axis = 3)
    except:
        merge8 = merge([conv2,up8],mode = 'concat', concat_axis = 3)
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8))
    conv8 = BatchNormalization()(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8))
 
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    try:
        merge9 = concatenate([conv1,up9],axis = 3)
    except:
        merge9 = merge([conv1,up9],mode = 'concat', concat_axis = 3)
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9))
    conv9 = BatchNormalization()(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation = 'softmax')(conv9)
 
    model = Model(inputs = inputs, outputs = conv10)
 
    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer = Adam(lr = learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #  如果有预训练的权重
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
 
    return model
if __name__ == '__main__':
    model=attunet()
    model.summary()