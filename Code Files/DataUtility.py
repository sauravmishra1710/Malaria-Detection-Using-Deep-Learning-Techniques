# Import global packages and libraries that are required 
# by the methods included in the file.

import numpy as np
import pandas as pd
import os
import random
from skimage import io
from skimage.transform import resize
import datetime
import glob
import random as rn
import gc

import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2

from scipy import stats
from statistics import mean 
from statistics import median 
import seaborn as sns

from concurrent import futures
import threading

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import Markdown
from enum import Enum


class IMAGE_AUGMENTATION_TYPE(Enum):
    
    '''
    An ENUM class to contain the enumeration values for the various augmentation techniques and their values 
    (the names as defined in the Keras - ImageDataGenerator class).
    '''
    
    HORIZONTAL_FLIP = 'horizontal_flip'
    VERTICAL_FLIP = 'vertical_flip'
    HORIZONTAL_SHIFT = 'horizontal_shift'
    VERTICAL_SHIFT = 'vertical_shift'
    CHANNEL_SHIFT = 'channel_shift_range'
    RANDOM_ROTATION = 'rotation_range'
    RANDOM_BRIGHTNESS = 'brightness_range'
    RANDOM_ZOOM = 'zoom_range'
    RANDOM_SHEAR = 'shear_range'

class RBC_CELL_TYPE(Enum):
    
    '''
    An ENUM class to contain the enumeration values for Red Blood Cell categories (Parasitized or Uninfected) 
    in this study.
    '''
    
    PARASITIZED = 'Parasitized'
    NONPARASITIZED = 'Uninfected'


class dataUtils():
    
    '''
    A Utility/Helper class that contains helper methods for some exploratry image based analysis.
    The methods majorly includes extracting the statistical level details of the images and plotting 
    various pre-processing and augmentation level transformations on sample images. 
    '''
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def PrintMarkdownText(textToDisplay):
        
        '''
        Purpose: 
            A static method to display markdown formatted output like bold, italic bold etc..

        Parameters:
            1. textToDisplay - the string message with formatting styles that is to be displayed

        Return Value: 
            NONE
        '''
        
        display(Markdown('<br>'))
        display(Markdown(textToDisplay))

    
    def GetLabelledMalarialData(self):
        
        '''
        Purpose: 
            Creates a dataframe with the filenames of all the malarial images and the 
            corresponding labels. The dataframe has 2 columns - 'filename' and 'label'

        Parameters: 
            NONE

        Return Value: 
            The computed malarial dataframe.
        '''
        
        parasitized_images = glob.glob('cell_images/Parasitized/*.png')
        non_parasitized_images = glob.glob('cell_images/Uninfected/*.png')
        len(parasitized_images), len(non_parasitized_images)

        malaria_df = pd.DataFrame({
            'filename': parasitized_images + non_parasitized_images,
            'label': ['Parasitized'] * len(parasitized_images) + ['NonParasitized'] * len(non_parasitized_images)
        })

        # Shuffle the rows in the dataset
        malaria_df = malaria_df.sample(frac=1, random_state=34).reset_index(drop=True)

        return malaria_df

   
    def GetImageDirectory(self, imageCategory):
        
        '''
        Purpose: 
            Gets the directory path for the category of image (Parasitized/Uninfected) in the dataset.

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            The image directory path.
        '''

        # Get the correct directory path based on the category of image
        if imageCategory == RBC_CELL_TYPE.PARASITIZED.value:
            imgDirectory = 'cell_images/Parasitized/*.png'
        else:
            imgDirectory = 'cell_images/Uninfected/*.png'

        return imgDirectory


    
    def GetAllImageShape(self, imageCategory):
        
        '''
        Purpose: 
            Extract the Image dimensions of the images in the dataset for the given imageCategory 
            (Parasitized/Uninfected). As the number of images are large, this method utilizes parallel 
            processing using the ThreadPoolExecutor for faster computation.

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            A list of dinemsions of all the images of the concerned imageCategory.
        '''
        
        images = []
        
        '''A nested function to get the image shape that is called parallely from the ThreadPoolExecutor
        Parameter:
            img - The image that is to be resized.'''
        def GetImageShape(img):
            return cv2.imread(img).shape

        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imageCategory)

        for img_path in glob.glob(imgDirectory):
            images.append(img_path)
        
        # https://docs.python.org/3/library/concurrent.futures.html for details on max_workers
        executer = futures.ThreadPoolExecutor(max_workers=None)
        all_image_dimension_map = executer.map(GetImageShape, [img for img in images])

        return list(all_image_dimension_map)

    
    def ResizeAllImages(self, imageList, newImageSize):
        
        '''
        Purpose: 
            Resizes all the images passed in to the new dimension defined in 'newImageSize'. 
            As the number of images are large, this method utilizes parallel processing using the 
            ThreadPoolExecutor for faster computation.

        Parameters:
            1. imageList - The list of all the images that are to be re-sized.
            2. newImageSize - The size to which the images have to be re-sized.

        Return Value: 
            List of the resized images.
        '''
        
        '''A nested function to resize the image to the specified new dimension 
        and is called parallely from the ThreadPoolExecutor.
        Parameter:
            img - The image that is to be resized.'''
        def ResizeImage(img):
            img = cv2.imread(img)
            img = cv2.resize(img, dsize=newImageSize, interpolation=cv2.INTER_CUBIC)
            img = np.array(img, dtype=np.float32)
            return img
        
        # https://docs.python.org/3/library/concurrent.futures.html for details on max_workers
        executer = futures.ThreadPoolExecutor(max_workers=None)
        train_data_map = executer.map(ResizeImage, [image for image in imageList])

        return np.array(list(train_data_map))
    
    
    
    def ReadAndDisplayInputImages(self, imageCategory, numImagesToDisplay):
        
        '''
        Purpose: 
            Read and display the first 5 images of the imageCategory (Parasitized/Uninfected) 
            in the dataset.

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.
            2. numImagesToDisplay - Total number of images to display.

        Return Value: 
            NONE
        '''
        
        images = []
        
        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imageCategory)

        # Read the first 5 images
        for img_path in glob.glob(imgDirectory):
            if len(images) < numImagesToDisplay:
                images.append(img.imread(img_path))

        # Display the images
        plt.figure(figsize=(20,10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.imshow(image)
            plt.axis('off')

    
            
    def DisplayAnnotatedImages(self, df, numImagesToDisplay):
        
        '''
        Purpose: 
            Display the given number of annotated images from the passed in dataframe of image 
            filenames and labels.

        Parameters:
            1. df - The dataset which contails the filenames and the corresponding labels.
            2. numImagesToDisplay - Total number of images to display.

        Return Value: 
            NONE
        '''
        
        images = []
        labels = []

        # Get the correct directory path based on the category of image
        # imgDirectory = self.GetImageDirectory(imageCategory)

        # Read the first 'numImagesToDisplay' images.
        for img_path in df.filename:
            if len(images) < numImagesToDisplay:
                # extract the corresponding image label
                label = df.loc[df['filename'] == img_path].label
                labels.append(label)
                images.append(img.imread(img_path))

        # Display the images
        plt.figure(figsize=(20,10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            plt.imshow(image)
            plt.title(labels[i].item())
            plt.axis('off')
    
    
  
    def ComputeAndPlotImageDimensionalStatistics(self, imgCategory):
        
        '''
        Purpose: 
            Computed and displays the dimensional statistics of the images in the directory and
            plots the distribution for the X and Y dimensional component.

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        dim_x = []
        dim_y = []
        
        allImageDims = self.GetAllImageShape(imgCategory)
        
        setOfUniqueDimensions = set(allImageDims)
        
        for shape in allImageDims:
            x,y,channel = shape
            dim_x.append(x)
            dim_y.append(y)


        f, ax = plt.subplots(1, 2)
        f.set_figwidth(10)

        sns.distplot(dim_x, kde=True, fit=stats.gamma, ax=ax[0]);
        sns.distplot(dim_y, kde=True, fit=stats.gamma, ax=ax[1]);

        ax[0].title.set_text('Distribution of X Dimension')
        ax[1].title.set_text('Distribution of Y Dimension')

        plt.show()
        
        print('Statistical Features - Image Dimension:')
        print('---------------------------------------')
        print('Max X Dimension:', max(dim_x))
        print('Min X Dimension:', min(dim_x))
        print('Mean X Dimension:', mean(dim_x))
        print('Median X Dimension:', median(dim_x))
        print('---------------------------------------')
        print('Max Y Dimension:', max(dim_y))
        print('Min Y Dimension:', min(dim_y))
        print('Mean Y Dimension:', mean(dim_y))
        print('Median Y Dimension:', median(dim_y))
        
        print('\nTotal # Images with Unique Dimensions:', len(setOfUniqueDimensions))

    
    
    def GetSampleImage(self, imgCategory):
        
        '''
        Purpose:
            Reads and returns the first image of the given category - (Parasitized/Uninfected).

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        images = []

        # Get the correct directory path based on the category of image
        imgDirectory = self.GetImageDirectory(imgCategory)

        # Read and return the first image in the directory.
        for img_path in glob.glob(imgDirectory):
            if len(images) < 1:
                images.append(img.imread(img_path))

        return images[0]
    

   
    def GetRandomImageName(self, imgCategory):
        
        '''
        Purpose: 
            Gets a random image of the given category - (Parasitized/Uninfected).

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            Full path of a random selected image.
        '''
        
        directoryPath ='cell_images/' + imgCategory + '/'
        all_images = os.listdir(directoryPath)
        imgIndex = random.randrange(0, len(all_images)+1) # adding 1 to include the last image index in random search
        full_img_path = directoryPath + all_images[imgIndex]
        return full_img_path

    
    def AugmentAndPlotData(self, imgCategory, augmentType):
        
        '''
        Purpose:
            Applies the concerned augmentation type on the image and display the original and 
            the augmentated images.

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.
            2. augmentType - The augmentation echnique that is to be applied on the image. 
                          See IMAGE_AUGMENTATION_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        strPlotTitle = ''
        datagen = None

        imgFileName = self.GetRandomImageName(imgCategory)

        # load the image
        img = load_img(imgFileName) #('cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png')

        # plot the original image
        plt.imshow(img)
        plt.suptitle('Original Image', y=0.94)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', 
                                   labelbottom='off', right='off', left='off', labelleft='off')
        # convert to numpy array
        data = img_to_array(img)

        # expand dimension to one sample
        samples = expand_dims(data, 0)

        # create image data augmentation generator based on the augmentation type passed
        if augmentType == IMAGE_AUGMENTATION_TYPE.HORIZONTAL_FLIP.value: #'horizontal_flip':
            datagen = ImageDataGenerator(horizontal_flip=True)
            strPlotTitle = 'Random Horizontally Flipped Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.VERTICAL_FLIP.value:
            datagen = ImageDataGenerator(vertical_flip=True)
            strPlotTitle = 'Random Vertically Flipped Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.HORIZONTAL_SHIFT.value:
            datagen = ImageDataGenerator(width_shift_range=[0.1, 0.2])
            strPlotTitle = 'Random Horizontally Shifted Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.VERTICAL_SHIFT.value:
            datagen = ImageDataGenerator(height_shift_range=0.3)
            strPlotTitle = 'Random Vertically Shifted Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.CHANNEL_SHIFT.value:
            datagen = ImageDataGenerator(channel_shift_range=40)
            strPlotTitle = 'Random Channel Shifted Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.RANDOM_ROTATION.value:
            datagen = ImageDataGenerator(rotation_range=90)
            strPlotTitle = 'Random Rotational Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.RANDOM_BRIGHTNESS.value:
            datagen = ImageDataGenerator(brightness_range=[0.5,1.0])
            strPlotTitle = 'Random Brightness Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.RANDOM_ZOOM.value:
            datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
            strPlotTitle = 'Random Zoom Augmented Images'

        elif augmentType == IMAGE_AUGMENTATION_TYPE.RANDOM_SHEAR.value:
            datagen = ImageDataGenerator(shear_range=30)
            strPlotTitle = 'Random Sheared Transformed Augmented Images'

        else:
            pass

        if datagen is not None:
            # prepare iterator
            imgIterator = datagen.flow(samples, batch_size=1)

            # generate samples and plot
            plt.figure(figsize=(8, 4))
            for i in range(8):
                # generate subplot
                plt.subplot(240 + 1 + i)

                # get the batch of images
                batch = imgIterator.next()

                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')

                # plot raw pixel data
                plt.imshow(image)
                plt.tick_params(axis='both', which='both', bottom='off', top='off', 
                                   labelbottom='off', right='off', left='off', labelleft='off')

            plt.suptitle(strPlotTitle, y=0.94)
            # show the figure
            plt.show()

        else:
            print("Datagen is " + str(datagen) +'.' + " Check the input for the augmentation type " +
                "and the implementation of ImageDataGenerator.")
            

            
    def PlotImageColorDistribution(self, imgCategory):
        
        '''
        Purpose: 
            Plots the RGB intensities and the histogram for a sample image
            of the given category - (Parasitized/Uninfected).

        Parameters:
            1. imgCategory - The category of the image (Parasitized/Uninfected). See RBC_CELL_TYPE ENUM.

        Return Value: 
            NONE
        '''
        
        f, axes = plt.subplots(1, 3)
        f.set_figwidth(16)

        imgFileName = self.GetRandomImageName(imgCategory)
        
        img = cv2.imread(imgFileName)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.imread('cell_images/Parasitized/C33P1thinF_IMG_20150619_114756a_cell_179.png', cv2.IMREAD_GRAYSCALE)

        color = ('r','g','b')

        # display the original image.
        axes[0].imshow(rgb_img)

        # extract and display the rgb intensities.
        for i,col in enumerate(color):
            histr = cv2.calcHist([rgb_img],[i],None,[256],[0,256])
            axes[1].plot(histr,color = col)

        # display the image histogram.
        axes[2].hist(img.ravel(),256,[0,256])

        # set the subplot title
        axes[0].title.set_text('Original Image')
        axes[1].title.set_text('RGB Intensity Distribution')
        axes[2].title.set_text('Histogram')

        plt.show()