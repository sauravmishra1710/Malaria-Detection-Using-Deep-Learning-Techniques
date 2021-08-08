import math
import os
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Lambda, concatenate
from tensorflow.keras.layers import Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from efficientnet.tfkeras import EfficientNetB0

class SnapshotEnsemble(Callback):
    
    '''
    The Snapshot Ensemble class to implement the snapshot callback
    to record the model snapshot (save the model weights) during the model training phase
    as the model visits several different local minima points.
    
    '''

    def __init__(self, folder_path, n_epochs, n_cycles=5, verbose=0):
        
        '''
        Purpose:
            Initialize the Snapshot Ensembling Framework

        Parameters:
            1. folder_path: Directory where the snapshots would be saved.
            2. n_epochs: The number of epochs the model would be trained.
            3. n_cycles: The number of training cycle for the cyclic LR scheduler.

        Returns:
            NONE.
        '''
        
        if n_cycles > n_epochs:
            raise ValueError('Number of cycles must be lower than number of epochs.')

        super(SnapshotEnsemble, self).__init__()
        self.verbose = verbose
        self.folder_path = folder_path
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.period = math.floor(self.n_epochs / self.n_cycles)
        self.digits = len(str(self.n_cycles))
        self.path_format = os.path.join(self.folder_path, 'model_snapshot_{}.h5')
        self.lrates = list()


    def on_epoch_end(self, epoch, logs=None):
        
        '''
        Purpose:
            Save model snapshots at the end of each cycle

        Parameters:
            1. epoch: The current training epoch number.
            2. logs

        Returns:
            NONE.
        '''
        
        # check if we can save model.
        # save if we are at the end of the cycle.
        if epoch == 0 or (epoch + 1) % self.period != 0: return

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        cycle = int(epoch / self.period)
        cycle_str = str(cycle+1)
        self.model.save(self.path_format.format(cycle_str), overwrite=True)

        # Update the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)

        if self.verbose > 0:
            print('\n\nEpoch %d: End of cycle %d , saving model snapshot to ' % ((epoch+1), (cycle+1)) + self.path_format.format(cycle_str) 
                  + '\n\n')


    def on_epoch_begin(self, epoch, logs=None):
        
        '''
        Purpose:
            Change the Learning Rate as per the Cosine Annealing Scheduler.

        Parameters:
            1. epoch: The current training epoch number.
            2. logs

        Returns:
            NONE.
        '''
        
        # calculate and set learning rate at the start of the epoch
        if epoch == 0: return

        lr = self.cosine_annealing(epoch)
        K.set_value(self.model.optimizer.lr, lr)
        
        # log learning rate value
        self.lrates.append(lr)

        if self.verbose > 0:
            print('\nEpoch %d: Cosine Annealing updating learning rate to %s.' % (epoch + 1, lr))


    def set_model(self, model):
        
        '''
        Purpose:
            Initialize the model.

        Parameters:
            1. model: The model set for snapshot ensembling.

        Returns:
            NONE.
        '''
        
        print("Inside Set Model")
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))
        print("self.base_lr",self.base_lr)


    def cosine_annealing(self, epoch):
        
        # calculate learning rate for epoch
        
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr
    
    @staticmethod
    def Custom_Model_With_Avg_Pooling(input_tensor, INPUT_SHAPE=(135, 135, 3), activation_mode='relu'):
        
        x = Conv2D(8, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation(activation_mode)(x)
        x = Dropout(0.5)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation(activation_mode)(x)
        x = Dropout(0.5)(x)

        x_out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_tensor, outputs=x_out, name="Custom_Model_With_Avg_Pooling")

        return model
    
    @staticmethod
    def GetEfficientNetB0Model(input_tensor, INPUT_SHAPE=(135, 135, 3), bTrainConvolutionBase=False):
        
        conv_base = EfficientNetB0(input_tensor=input_tensor, weights="imagenet", include_top=False)

        if bTrainConvolutionBase == False:
            for layer in conv_base.layers:
                layer.trainable = False

        x = conv_base.output
        x = GlobalAveragePooling2D(name="GAP")(x)
        x = Dropout(0.5, name="dropout_out")(x)

        x_out = Dense(1, activation="sigmoid", name="fc_out")(x)

        EfficientNetB0_TL_Model = Model(inputs=input_tensor, outputs=x_out, name="TL_Ensemble_EfficientNet_B0")

        return EfficientNetB0_TL_Model
    
    @staticmethod
    def Load_Ensembled_Model(snapshot_dir, input_size = (135, 135, 3)):
    
        print('\nEnsembling Snapshot Models\n')
        print('Loading Snapshots...')

        paths = glob.glob(os.path.join(snapshot_dir, '*.h5'))
        for path in paths:
            print('>>> Loaded -- ', path)

        x_in = Input(shape=input_size)
        outputs = []

        for i, path in enumerate(paths):
            m = SnapshotEnsemble.GetEfficientNetB0Model(x_in, bTrainConvolutionBase=True)
            # m = SnapshotEnsemble.Custom_Model_With_Avg_Pooling(x_in)
            # m = SnapshotEnsemble.Build_malaria_Inception_Network(x_in, bTrainConvolutionBase=True)

            # rename the layers with _i 'coz when trying to ensemble snapshots of a pre-trained network
            # the function Model might throw a runtime error as below - 
            # The name <"layer_name"> is used <n> times in the model. All layer names should be unique.
            for layer in m.layers:
                layer._name = layer.name + "_" + str(i)

            m.load_weights(path)
            outputs.append(m.output)
            m = None

        shape = outputs[0].get_shape().as_list()

        x_out = Lambda(lambda x: K.mean(K.stack(x, axis=0), axis=0), output_shape=lambda _: shape)(outputs)

        ensembled_model = Model(inputs=x_in, outputs=x_out)
        ensembled_model.compile(optimizer='adam',
                                loss='binary_crossentropy',
                                metrics=['accuracy'])

        print('\n')
        return ensembled_model
    
    @staticmethod
    def get_inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)

        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out
    
    @staticmethod
    def Build_malaria_Inception_Network(input_tensor):
        
        # define model input
        # visible = input_tensor #Input(shape = INPUT_SHAPE)
        
        # add inception block 1
        layer = SnapshotEnsemble.get_inception_module(input_tensor, 64, 96, 128, 16, 32, 32)
        
        # add inception block 1
        layer = SnapshotEnsemble.get_inception_module(layer, 128, 128, 192, 32, 96, 32)

        gap = GlobalAveragePooling2D(data_format='channels_last')(layer)
        dense = Dense(1024, activation='relu')(gap)
        dropout = Dropout(0.5)(dense)
        out  = Dense(1, activation='sigmoid')(dropout)

        # create model
        model = Model(inputs=input_tensor, outputs=out, name='malaria_inception_network')
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        # summarize model
        # print(model.summary())
        return model