from sklearn.utils import shuffle
from keras.utils import np_utils
import cv2
import numpy as np

from DataUtility import RBC_CELL_TYPE

class CustomDataGenerator():
    
    '''
    Generate batches of tensor image data with basic image pre-processing. 
    This helps in generting and feeding the images to the model batch-wise 
    rather than storing all the images in the memory.
    
    '''
    
    def Build_Image_Samples(self, image_files, labels):
    
        '''
        Purpose:
            Prepare the image samples for the generator.

        Parameters:
            1. image_files: The list of the image file names.
            2. labels: List of the corresponding image labels.

        Returns:
            The image zipped sample object containing the image 
            files and the corresponding labels.
        '''
        samples=[]

        for image, label in zip(image_files, labels):
            samples.append([image, label])

        return samples

    def transform_label(self, label):
        '''
        Purpose:
            Transform the label into binarized (0 , 1) format.
            PARASITIZED - 1
            NON-PARASITIZED - 0

        Parameters:
            1. label: The corresponding image label.

        Returns:
            The encoded label (0 or 1)
        '''
        if label == RBC_CELL_TYPE.PARASITIZED.value:
            enc_label = 1
        else:
            enc_label = 0

        return enc_label


    def preprocess_input(self, image, label, resize_dims = (135, 135)):
        '''
        Purpose:
            Pre process the image - apply any image processing prior 
            to yeilding the image for the generator.

        Parameters:
            1. image: The image object.
            2. label: The corresponding image label.
            3. resize_dims: The dimensions to which the image is to be resized. 
                            The default value is (135, 135)

        Returns:
            1. image: The processed image.
            2. label: The encoded label (0 or 1).
        '''
        # Resize and standardize the image
        image = cv2.resize(image, resize_dims)
        image = image/255

        # get encoded label
        label = self.transform_label(label)

        return image, label
    
    
    def CustomImageDataGenerator(self, samples, batch_size = 64, shuffle_data = True, resize_dims = (135, 135)):
       
        '''
            Purpose:
                Yields the next training batch.

            Parameters:
                1. samples: The image zipped sample object containing the image files and the corresponding labels.
                2. batch_size: The number of training examples utilized in one iteration
                3. shuffle_data:  Whether to shuffle the data. Default: True. 
                4. resize_dims: The dimensions to which the image is to be resized. The default value is (135, 135)

            Returns:
                Yields the next sequence of batched training/validation data.
            '''
        
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
            samples = shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                # Get the training batch samples
                batch_samples = samples[offset:offset+batch_size]

                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []

                # For each example
                for batch_sample in batch_samples:
                    # Extract the image & label
                    img_name = batch_sample[0]
                    label = batch_sample[1]

                    img =  cv2.imread(img_name)
                    # preprocess the image
                    img,label = self.preprocess_input(img, label, resize_dims)

                    X_train.append(img)
                    y_train.append(label)

                # Convert to numpy arrays
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # yield the next training batch            
                yield X_train, y_train