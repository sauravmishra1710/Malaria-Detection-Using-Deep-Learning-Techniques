# Malaria Parasite Detection using Efficient Neural Ensembles
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/malaria-parasite-detection-using-efficient/medical-image-classification-on-malaria)](https://paperswithcode.com/sota/medical-image-classification-on-malaria?p=malaria-parasite-detection-using-efficient)

This work is published as part of the [Journal of Electronics, Electromedical Engineering, and Medical Informatics](https://jeeemi.org/index.php/jeeemi) and can be accessed online at the [Journal Page](https://doi.org/10.35882/jeeemi.v3i3.2). Please [cite](#citeAs) the work if you find these codes useful for your research. 

This work is an open-access and licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa] [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa] 

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

**<ins>LINK TO ABSTRACT</ins>:** http://jeeemi.org/index.php/jeeemi/article/view/147 </br>
**<ins>LINK TO FULL PDF</ins>:** http://jeeemi.org/index.php/jeeemi/article/view/147/75

## ABSTRACT

Malaria, a life threatening disease caused by the bite of the Anopheles mosquito infected with the parasite of genus Plasmodium, has persisted as a major burden towards mankind and healthcare for years leading to approximately 400,000 deaths globally every year. The traditional diagnosis process for malaria involves examining the blood smear slide under the microscope. This is a very time consuming process and also requires pathologists to be highly skilled for performing the analysis. Timely diagnosis, availability of robust diagnostic facilities, and skilled laboratory technicians are very much vital to bring down the high mortality rate. 

This study aims to build an efficient system by applying ensemble techniques based on  deep learning to automate the detection of the parasite using whole slide images of thin blood smears. A time efficient technique known as the Snapshot Ensemble [(Huang et. al.,)](https://arxiv.org/abs/1704.00109)  will be used in this study. Snapshot Ensemble is a technique which at the cost of training one single neural network would yield multiple snapshots or weak learners that can later be combined to form one strong model. All the experimental models were evaluated against the following metrics - F1 score, Accuracy, Precision, Recall, Mathews Correlation Coefficient (MCC), Area Under the Receiver Operating Characteristics (AUC-ROC) and the Area under the Precision Recall curve (AUC-PR). The snapshot ensemble model created by combining the snapshots of the EfficientNet-B0 pre-trained model outperformed every other model achieving a f1 score - 99.37%, precision - 99.52% and recall - 99.23%. The results show the potential of  model ensembles which combine the predictive power of multiple weak learners to create a single strong and efficient model that is better equipped to handle the real world data. Also, snapshot ensemble proved to be efficient with a limited training budget.

The GradCAM [(Selvaraju et. al.,)](https://arxiv.org/abs/1610.02391) experiment displayed the gradient activation maps of the last convolution layer to visually explicate where and what a model sees in an image to classify them into a particular class. The models in this study correctly activate the stained parasitic region of interest in the thin blood smear images. Such visuals make the model more transparent, explainable, and trustworthy which are very much essential for deploying AI based models in the healthcare network.


## PROGRAMMING RESOURCES

All programming work in the study is done on a cloud based system with 

- CUDA enabled Nvidia Tesla K80 GPU, 4 core CPUs, 20 GB RAM. 
- Python 3.6 using  the  web-based Anaconda Jupyter  environment.  
- The deep  learning  models  are  created  using  Keras  library  with Tensorflow 2.2 backend enabled with GPU acceleration.
- All the image processing and computer vision tasks are carried out  using  the  Open  Source  Computer  Vision  (OpenCV) library.


## RESULTS

### EFFICIENTNET-B0 SNAPSHOTS

The callback settings for cosine annealing was kept for 50 epochs and number of cycles set to 5.  The  snapshots  were  recorded  each  at  the  end of  10th, 20th,  30th,  40th  and  50th  epochs  respectively.  Each  of  the snapshot  was  evaluated  on  the  test  data  and  the  results  are summarized in the [table 1](#tab1) below. 

#### <h4 id="tab1">Table 1 - Snapshots Test Evaluation:</h4>
![Table 1 - Snapshots Test Evaluation](https://github.com/sauravmishra1710/Malaria-Detection-Using-Deep-Learning-Techniques/blob/main/Test%20Evaluation%20Metrics/Snapshot%20Ensemble/EfficientNet-B0/SnapshotsEvaluation.PNG)

- Ensemble  1 combined every snapshots [1, 2,  3,  4,  5]
- Ensemble 2 combined  snapshots  [2,  3,  4,  5]
- Ensemble 3  combined snapshot [3, 4, 5]
- Ensemble 4 combined snapshot [3, 4]
- Ensemble 5 combined snapshot [4, 5]

Five different combinations of the snapshots were tested to create the ensembled model in the following combination strategies. Each of the ensembled model  was  evaluated  on  the  test  data  and  the  results  are summarized in [table 2](#tab2) below.

#### <h4 id="tab2">Table 2 - Ensembles Test Evaluation:</h4>
![Table 1 - Snapshots Test Evaluation](https://github.com/sauravmishra1710/Malaria-Detection-Using-Deep-Learning-Techniques/blob/main/Test%20Evaluation%20Metrics/Snapshot%20Ensemble/EfficientNet-B0/EnsembleEvaluation.PNG)

## SUMMARY

- Snapshot Ensemble, enables to create high performance ensembled models by training only a single neural network. This approach not only removes the need of high efficiency computational resources but also saves valuable amount of time required to train multiple large models to combine and create ensembles. 

- GradCAM experiment shows where exactly a model looks in the  image to arrive at a decision. We see the model correctly activates the region around the parasite in the parasitized cells and utilizes it to differentiate from the unparasitized cells. Such visuals make the models more transparent, explainable, and trustworthy which are very muchnessential for deploying AI based models in the healthcare network.

## Papers With Code

Link: https://paperswithcode.com/paper/malaria-parasite-detection-using-efficient

<h2 id="citeAs">HOW TO CITE</h2>

If you find this work helpful for your study, please cite the paper as follows -</br>

Mishra, S. 2021. Malaria Parasite Detection using Efficient Neural Ensembles. Journal of Electronics, Electromedical Engineering, and Medical Informatics. 3, 3 (Oct. 2021), 119-133. DOI:https://doi.org/10.35882/jeeemi.v3i3.2.
