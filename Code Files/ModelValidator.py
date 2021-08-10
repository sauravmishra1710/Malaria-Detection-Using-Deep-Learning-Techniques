
# Import Packages and Libraries
from IPython.display import Markdown
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class ValidateModel():
    
    def __init__(self):
        
        '''
        The Model Validator class to validate and compute the various classification metrics and scores for
        the models. The module returns a consolidated classification report with the following metric scores.

            1. Confusion Matrix
            2. Classification Report
            3. F1 Score
            4. Accuracy
            5. Mathews Correlation Coefficient (MCC)
            6. Precision
            7. Recall
            8. AUROC Score - Area Under the Receiver Operating Characteristic Curve
            9. AUC-PR Score - Area Under the Precision Recall Curve.
            10. Plot for AUROC Curve - Area Under the Receiver Operating Characteristic Curve
            11. Plot for AUC-PR Curve - Area Under the Precision Recall Curve.

        '''
        
        pass
    
    
    def PrintMarkdownText(self, textToDisplay):
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
        
    def PrepareImageForPrediction(self, image):
        '''
        Resizes and expands dimension

        Input: image: a 3-channel image as input

        Returns a rank-4 tensor, since the network accepts batches of images
        One image corresponds to batch size of 1
        '''
        img_4d = np.expand_dims(image, axis=0)  # rank 4 tensor for prediction
        return img_4d
    
    
    def plot_confusion_matrix(self, y_test, y_preds):
        
        '''
        Purpose: 
            Plot the confusion matrix on y_test and y_pred for the model.

        Parameters:
            1. y_test - The Ground Truth for each test image.
            2. y_pred - The Predicted label for each image.

        Return Value: 
            NONE.
        '''
        
        conf_matx = confusion_matrix(y_test, y_preds)
        sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="GnBu")
        plt.show()
    
    # Custom Function to get Scores and plots...
    def Generate_Model_Test_Classification_Report(self, y_test, y_pred, model_name=""):
        
        '''
        Purpose: 
            Generate the consolidated test classification report. 
            The report consists of the following classification results & metrics -
                1. Confusion Matrix
                2. Classification Report
                3. F1 Score
                4. Accuracy
                5. Mathews Correlation Coefficient (MCC)
                6. Precision
                7. Recall
                8. AUROC Score - Area Under the Receiver Operating Characteristic Curve
                9. AUC-PR Score - Area Under the Precision Recall Curve.
                10. AUROC Curve - Area Under the Receiver Operating Characteristic Curve
                11. AUC-PR Curve - Area Under the Precision Recall Curve.

        Parameters:
            1. y_test - The Ground Truth for each test image.
            2. y_pred - The Predicted label for each image.
            3. model_name - Model Name

        Return Value: 
            NONE.
        '''
        
        # Report Title & Classification Mterics Abbreviations...
        fig, axes = plt.subplots(3, 1, figsize = (8, 3))
        axes[0].text(9, 1.8, "CONSOLIDATED MODEL TEST REPORT", fontsize=30, horizontalalignment='center', 
                     color='DarkBlue', weight = 'bold')

        axes[0].axis([0, 10, 0, 10])
        axes[0].axis('off')
        
        axes[1].text(9, 4, "Model Name: " + model_name, style='italic', 
                             fontsize=18, horizontalalignment='center', color='DarkOrange', weight = 'bold')

        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')

        axes[2].text(0, 4, "* 1 - Parasitized\t\t\t\t\t\t\t * 0 - Non Parasitized\n".expandtabs() +
                     "* MCC - Matthews Correlation Coefficient\t\t* AUC - Area Under The Curve\n".expandtabs() +
                     "* ROC - Receiver Operating Characteristics     " + 
                     "\t* AUROC - Area Under the Receiver Operating    Characteristics".expandtabs(), 
                     style='italic', fontsize=10, horizontalalignment='left', color='orangered')

        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')
        
        scores = []
        metrics = ['F1       ', 'MCC      ', 'Precision', 'Recall   ', 'Accuracy ',
                   'AUC_ROC  ', 'AUC_PR   ']

        # Plot ROC and PR curves using all models and test data...
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        precision, recall, th = precision_recall_curve(y_test, y_pred)

        # Calculate the individual classification metic scores...
        model_f1_score = f1_score(y_test, y_pred)
        model_matthews_corrcoef_score = matthews_corrcoef(y_test, y_pred)
        model_precision_score = precision_score(y_test, y_pred)
        model_recall_score = recall_score(y_test, y_pred)
        model_accuracy_score = accuracy_score(y_test, y_pred)
        model_auc_roc = auc(fpr, tpr)
        model_auc_pr = auc(recall, precision)

        scores.append([model_f1_score,
                       model_matthews_corrcoef_score,
                       model_precision_score,
                       model_recall_score,
                       model_accuracy_score,
                       model_auc_roc,
                       model_auc_pr])

        sampling_results = pd.DataFrame(columns = ['Classification Metric', 'Score Value'])
        for i in range(len(scores[0])):
            sampling_results.loc[i] = [metrics[i], scores[0][i]]

        sampling_results.index = np.arange(1, len(sampling_results) + 1)

        class_report = classification_report(y_test, y_pred)
        conf_matx = confusion_matrix(y_test, y_pred)

        # Display the Confusion Matrix...
        fig, axes = plt.subplots(1, 3, figsize = (20, 4))
        sns.heatmap(conf_matx, annot=True, annot_kws={"size": 16},fmt='g', cbar=False, cmap="GnBu", ax=axes[0])
        axes[0].set_title("1. Confusion Matrix", fontsize=21, color='darkgreen', weight = 'bold', 
                          style='italic', loc='left', y=0.80)
        
        # Classification Metrics
        axes[1].text(5, 1.8, sampling_results.to_string(float_format='{:,.4f}'.format, index=False), style='italic', 
                     fontsize=20, horizontalalignment='center')
        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')
        axes[1].set_title("2. Classification Metrics", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=0.80)

        # Classification Report
        axes[2].text(0, 1, class_report, style='italic', fontsize=20)
        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')
        axes[2].set_title("3. Classification Report", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=0.80)

        plt.tight_layout()
        plt.show()

        # AUC-ROC & Precision-Recall Curve
        fig, axes = plt.subplots(1, 2, figsize = (14, 4))

        axes[0].plot(fpr, tpr, label = f"auc_roc = {model_auc_roc:.3f}")
        axes[1].plot(recall, precision, label = f"auc_pr = {model_auc_pr:.3f}")

        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].legend(loc = "lower right")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("4. AUC - ROC Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=1, y=1.05)

        axes[1].legend(loc = "lower left")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("5. Precision - Recall Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=3, y=1.05)

        plt.subplots_adjust(top=0.95) 
        plt.tight_layout()
        plt.show()

    
    def PlotModelPredictionsOnRandomTestImages(self, X_test, y_test, y_pred):
        
        '''
        Purpose: 
            Display Predictions of random images from the test set with the probability score.

        Parameters:
            1. X_test - The test image set.
            2. y_test - The Ground Truth for each test image.
            3. y_pred - The Predicted label for each image.

        Return Value: 
            NONE.
        '''
        
        self.PrintMarkdownText("***Model test results of 16 random test images...***")

        fig = plt.figure(figsize=(16, 16))
        columns = 4
        rows = 4
        random_number = np.random.randint(0,X_test.shape[0]-17)
        for i in range(1, columns*rows +1):
            infection_severity = None
            fig.add_subplot(rows, columns, i)
            plt.imshow(X_test[i+random_number])
            ground_truth = ['Non Parasitized', 'Parasitized']

            plt.axis('off')
            prob_scr = round(y_pred[i+random_number][0], 2) * 100
            # 1 - Parasitized & 0 - Non Parasitized 
            # ==> If y_test[i] is 1, we retrieve ground_truth[1] which is 'Parasitized' &
            #     If y_test[i] is 0, we retrieve ground_truth[0] which is 'Non Parasitized'
            if prob_scr == 0:
                infection_severity = 'Uninfected'
                pass
            elif prob_scr > 0 and prob_scr <= 5:
                infection_severity = 'Very Low'
                pass
            elif prob_scr > 5 and prob_scr <= 25:
                infection_severity = 'Low'
                pass
            elif prob_scr > 25 and prob_scr <= 50:
                infection_severity = 'Moderate'
                pass
            else:
                infection_severity = 'High'
            
            plt.title('Infection Probability: {:.1%}\n Infection Severity: {}'
                      .format(float(y_pred[i+random_number]), infection_severity))


        fig.tight_layout()   
        plt.show()
        
    
    def GetModelPredictions(self, model, X_test):

        '''
        Purpose: 
            Get the model prediction scores.
            1. The probability score and 2. The y_pred_binary score (0 or 1) depending on the threshold of 0.5

        Parameters:
            1. model - The model for which the predictions are to be made.
            2. X_test - The test image set.

        Return Value: 
            1. predicted_probabilities - The probability score array.
            2. prediction_binaries - The image class (0 or 1) depending on the probability score and threshold of 0.5
        '''

        threshold = 0.50

        predicted_probabilities = model.predict(X_test)
        prediction_binaries = [int(pred > threshold) for pred in predicted_probabilities]

        return predicted_probabilities, prediction_binaries
    