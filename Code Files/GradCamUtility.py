'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DESCRIPTION: Grad-CAM class activation visualization.
             A Utility/Helper class that contains helper methods for visualizing the class level activations of 
             a Convolotional Neural Network. Grad-CAM visualizations help us know which regions, patterns of 
             the image the neural network is looking at, and activating the region around those patterns. To 
             visualize the activation maps, we would need the output of the LAST CONVOLUTIONAL LAYERS
             and the final CLASSIFICATION LAYERS.
             
REFERENCE:   (François Chollet, 2020)
             François Chollet, (2020) Grad-CAM class activation visualization. 
             [online] Available at: https://keras.io/examples/vision/grad_cam/ [Accessed 20 Jun. 2020].

WEBSITE:     https://keras.io/examples/vision/grad_cam/
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Import Libraries
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCamUtils():

    def __init__(self):
        pass


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        Function to extract the image array.

    Parameters:
        1. img_path - the path of the image for which the array is required.
        2. size - the preferred size of the image to be read.

    Return Value: 
        img_array - The array representation (in terms of the batch - 
                    (1, size_x, size_y, channel)) of the image.
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def GetImageArrayInBatch(self, img_path, size):

        img = keras.preprocessing.image.load_img(img_path, target_size=size)

        # Get the image array
        img_array = keras.preprocessing.image.img_to_array(img)

        # We add a dimension to transform our array into a "batch"
        # of size (1, size_x, size_y, channel)
        img_array = np.expand_dims(array, axis=0)
        return img_array

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        Get the class level activations heatmap for the CNN network. The activation maps help us 
        identify & localize the regions, patterns of the image the neural network looks at, and 
        activates the region around the patterns. To visualize the activation maps, this function 
        works on the output of the LAST CONVOLUTIONAL LAYER and the final CLASSIFICATION LAYERS.

    Parameters:
        1. img_array - the array representation of the image for which the activation maps are to be 
                       visualized.
        2. model - the CNN model whose activations are to be analyzed.
        3. last_conv_layer_name - name of the last convolutional layer of the model.
        4. classifier_layer_names - the final classification layers.

    Return Value: 
        heatmap - The heatmap showing the more active regions the CNN looked at in deciding the class
                  for the image.
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def ComputeGradCAMHeatmap(self, img_array, model, last_conv_layer_name, classifier_layer_names):
        
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        
        # np.maximum(heatmap, 0) - This is the ReLU operation to ensure we consider
        # only those features that tend to have a positive effect and increase the
        # probability score for the particular class.
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        
        return heatmap
    
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        Display the heatmap - the class level activations.

    Parameters:
        1. heatmap - The class activation heatmap for the concerned image.
        
    Return Value: 
        NONE
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def DisplayHeatMap(self, heatmap):
        # Display heatmap
        plt.matshow(heatmap)
        plt.axis('off')
        plt.show()


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        Get the super imposed or the blended version of the image comprising of the class activations
        heatmap and the original image.

    Parameters:
        1. heatmap - The class activation heatmap for the concerned image.
        2. img - The original image for which the activation image is being calculated.
        
    Return Value: 
        superImposedImage - The blended version of the original image and the corresponding class 
                            activation map.
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def GetSuperImposedCAMImage(self, heatmap, img):
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superImposedImage = cv2.addWeighted(jet_heatmap, 0.2, img, 0.8, 0.0)
        
        return superImposedImage

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Purpose: 
        Display the original image and the corresponding class activation blended image.

    Parameters:
        1. image - The original image for which the activation image is being calculated.
        2. superimposed_img - The blended version of the original image and the corresponding class 
                              activation map. 
        
    Return Value: 
        NONE
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def DisplaySuperImposedImages(self, image, heatmap,superimposed_img):
        fig, ax = plt.subplots(1, 3, figsize=(8, 12))

        ax[0].imshow(image)
        ax[1].imshow(heatmap)
        ax[2].imshow(superimposed_img)

        ax[0].title.set_text('Original Image')
        ax[1].title.set_text('Class Activation Heatmap')
        ax[2].title.set_text('Class Activation Blended Image')

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')