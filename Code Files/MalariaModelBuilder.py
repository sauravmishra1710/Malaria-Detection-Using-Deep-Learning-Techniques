# Import reqiured pckages and libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Activation, Dropout, ZeroPadding3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

# Import the pre-trained models
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB4
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from IPython.display import Markdown


class ModelBuilder():
    
    '''
    The Model Builder class to design the different models
    to be experimented for the study.
    
    '''
    
    def __init__(self):
        pass
    
    def Custom_Model_With_Avg_Pooling(self, INPUT_SHAPE=(135, 135, 3), activation_mode='relu'):
        
        model = Sequential(name='Custom_Model_With_Avg_Pooling')

        model.add(Conv2D(8, kernel_size=(1, 1), padding='same', input_shape=INPUT_SHAPE))
        model.add(Activation(activation_mode))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(AveragePooling2D(pool_size=(2, 2)))
            
        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation(activation_mode))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation(activation_mode))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model
    
    
    
    def Custom_Model_With_Max_Pooling(self, INPUT_SHAPE=(135, 135, 3), activation_mode='relu'):
        
        model = Sequential(name='Custom_Model_With_Max_Pooling')

        model.add(Conv2D(8, kernel_size=(1, 1), padding='same', input_shape=INPUT_SHAPE))
        model.add(Activation(activation_mode))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation(activation_mode))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation(activation_mode))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation(activation_mode))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model
    
    def BuildCustomModelWith_GAP_Layer(self, INPUT_SHAPE=(135, 135, 3)):

        model = Sequential(name = 'Custom_Model_with_GAP_Layer')

        model.add(Conv2D(32, (3,3), padding='same',input_shape = INPUT_SHAPE))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(GlobalAveragePooling2D(data_format='channels_last'))

        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model
    

    def GetEfficientNetB0Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):
        
        conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False
        
        dropout_rate = 0.3
        EfficientNetB0_TL_Model = Sequential(name='Transfer_Learning_EfficientNet_B0')
        EfficientNetB0_TL_Model.add(conv_base)

        EfficientNetB0_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            EfficientNetB0_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        EfficientNetB0_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        EfficientNetB0_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(EfficientNetB0_TL_Model.summary())
        return EfficientNetB0_TL_Model
    
    def GetEfficientNetB4Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):
        
        conv_base = EfficientNetB4(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_EfficientNetB4_Retrain_Conv_Base"
        else:
            model_name = "Transfer_Learning_EfficientNetB4"

        dropout_rate = 0.5
        EfficientNetB4_TL_Model = Sequential(name=model_name)
        EfficientNetB4_TL_Model.add(conv_base)

        EfficientNetB4_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            EfficientNetB4_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        EfficientNetB4_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        EfficientNetB4_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(EfficientNetB4_TL_Model.summary())
        return EfficientNetB4_TL_Model
    
    def GetInceptionv3Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.3
        conv_base = InceptionV3(input_shape = INPUT_SHAPE, include_top = False, weights = 'imagenet')

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False
                
        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_Inception_v3_Retrain_Conv_Base"
        else:
            model_name = "Transfer_Learning_Inception_v3"

        Inceptionv3_TL_Model = Sequential(name='Transfer_Learning_Inception_v3')
        Inceptionv3_TL_Model.add(conv_base)

        Inceptionv3_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            Inceptionv3_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        Inceptionv3_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        Inceptionv3_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(Inceptionv3_TL_Model.summary())
        return Inceptionv3_TL_Model
    
    
    def GetVGG19Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):
    
        conv_base = VGG19(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False
                
        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_VGG19_Retrain_Conv_Base"
        else:
            model_name = "Transfer_Learning_VGG19"

        VGG19_TL_Model = Sequential(name=model_name)
        VGG19_TL_Model.add(conv_base)

        VGG19_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        VGG19_TL_Model.add(Dropout(0.5, name="dropout_out"))

        VGG19_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        VGG19_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(VGG19_TL_Model.summary())
        return VGG19_TL_Model
    
    
    def GetXceptionModel(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.3
        conv_base = Xception(input_shape = INPUT_SHAPE, include_top=False, weights = 'imagenet')

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_Xception_Retrain_Conv_Base"
        else:
            model_name = "Transfer_Learning_Xception"

        Xception_TL_Model = Sequential(name='Transfer_Learning_Xception')
        Xception_TL_Model.add(conv_base)

        Xception_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            Xception_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        Xception_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        Xception_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(Xception_TL_Model.summary())
        return Xception_TL_Model
    
    def GetNasNetMobileModel(self, INPUT_SHAPE=(224, 224, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.3
        conv_base = NASNetMobile(input_shape = INPUT_SHAPE, include_top=False, weights = 'imagenet')

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_NasNetMobile_Retrain_Base"
        else:
            model_name = "Transfer_Learning_NasNetMobile"

        NASNetMobile_TL_Model = Sequential(name=model_name)
        NASNetMobile_TL_Model.add(conv_base)

        NASNetMobile_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            NASNetMobile_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        NASNetMobile_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        NASNetMobile_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(NASNetMobile_TL_Model.summary())
        return NASNetMobile_TL_Model
    
    
    def GetInceptionResNetV2Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.5
        conv_base = InceptionResNetV2(input_shape = INPUT_SHAPE, include_top=False, weights = 'imagenet')


        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        if bTrainConvolutionBase == True:
            model_name = "Transfer_Learning_InceptionResNetV2_Retrain_Conv_Base"
        else:
            model_name = "Transfer_Learning_InceptionResNetV2"

        InceptionResNetV2_TL_Model = Sequential(name=model_name)
        InceptionResNetV2_TL_Model.add(conv_base)

        InceptionResNetV2_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            InceptionResNetV2_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        InceptionResNetV2_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        InceptionResNetV2_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(InceptionResNetV2_TL_Model.summary())
        return InceptionResNetV2_TL_Model
    
    

    def GetDenseNet121Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.5
        conv_base = DenseNet121(input_shape = INPUT_SHAPE, include_top=False, weights = 'imagenet')

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        DenseNet121_TL_Model = Sequential(name='Transfer_Learning_DenseNet121')
        DenseNet121_TL_Model.add(conv_base)

        DenseNet121_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            DenseNet121_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        DenseNet121_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        DenseNet121_TL_Model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])

        print(DenseNet121_TL_Model.summary())
        return DenseNet121_TL_Model
    
    

    def GetResNet50V2Model(self, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):

        dropout_rate = 0.5
        conv_base = ResNet50V2(input_shape = INPUT_SHAPE, include_top=False, weights = 'imagenet')

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        ResNet50V2_TL_Model = Sequential(name='Transfer_Learning_ResNet50V2')
        ResNet50V2_TL_Model.add(conv_base)

        ResNet50V2_TL_Model.add(GlobalAveragePooling2D(name="GAP"))

        if dropout_rate > 0:
            ResNet50V2_TL_Model.add(Dropout(dropout_rate, name="dropout_out"))

        ResNet50V2_TL_Model.add(Dense(1, activation="sigmoid", name="fc_out"))

        ResNet50V2_TL_Model.compile(optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])

        print(ResNet50V2_TL_Model.summary())
        return ResNet50V2_TL_Model
    
    
    def Custom_Model_With_Gaussian_Noise_Regularizer(self, INPUT_SHAPE=(135, 135, 3), noise_std_dev=0.02):
        
        model_name = "Gaussian_Noise_Regularized_Model_Noise_Level_" + str(noise_std_dev)
        model = Sequential(name=model_name)
        
        model.add(GaussianNoise(noise_std_dev, input_shape=INPUT_SHAPE, name='GaussianNoise'))
        
        model.add(Conv2D(8, kernel_size=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model
    
    
    def Custom_Model_With_Gaussian_Noise_Regularizer_In_Hidden_Layers(self, INPUT_SHAPE=(135, 135, 3)):
        
        model = Sequential(name='Custom_Model_With_Gaussian_Noise_Regularizer')
        
        model.add(GaussianNoise(0.02, input_shape=INPUT_SHAPE, name='GaussianNoise'))
        
        model.add(Conv2D(8, kernel_size=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(GaussianNoise(0.02))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(GaussianNoise(0.02))

        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(GaussianNoise(0.02))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GaussianDropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GaussianDropout(0.5))

        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())
        return model
    
    
    def get_inception_module(self, input_layer, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # reference - 
        # https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(input_layer)

        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(input_layer)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(input_layer)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_layer)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

        # concatenate the output of the convolutional layers and pass on
        # to the next layers
        output_layer = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return output_layer
    
    def Build_malaria_Inception_Network(self, INPUT_SHAPE=(135, 135, 3)):
        
        # define model input
        visible = Input(shape = INPUT_SHAPE)
        
        # add inception block 1
        layer = self.get_inception_module(visible, 64, 96, 128, 16, 32, 32)
        
        # add inception block 1
        layer = self.get_inception_module(layer, 128, 128, 192, 32, 96, 32)

        gap = GlobalAveragePooling2D(data_format='channels_last')(layer)
        dense = Dense(1024, activation='relu')(gap)
        dropout = Dropout(0.5)(dense)
        out  = Dense(1, activation='sigmoid')(dropout)

        # create model
        model = Model(inputs=visible, outputs=out, name='malaria_inception_network')
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        # summarize model
        print(model.summary())
        return model