**Semantic Segmentation using U-Net**

This repository contains code for performing semantic segmentation using the U-Net architecture on the LGG-MRI dataset. The U-Net model is a popular choice for image segmentation tasks, and this implementation aims to segment brain tumor regions from MRI scans.

**Introduction**

Semantic segmentation is a computer vision task that involves dividing an image into meaningful segments and assigning each pixel a corresponding label. In this project, we use the U-Net architecture to perform semantic segmentation of brain tumor regions in MRI scans. U-Net is a convolutional neural network (CNN) architecture that has been widely used for image segmentation tasks due to its effective feature extraction and upsampling capabilities.

**Dataset**

The dataset used in this project is the LGG-MRI dataset, which consists of MRI scans and corresponding tumor masks. The data has been preprocessed and resized to (224, 224) for training the U-Net model.The image of brain MRI and corrresponding mask is shown in figure below:

![download](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/2fa25f15-0e12-48ce-b6b1-92c0a669399a)

![download (1)](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/62c343c7-89d3-4d2a-92de-2858fc324832)

![download (2)](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/4b382d75-1a2c-4a39-a75a-3807fdc81095)


**Model Architecture**

The U-Net model used in this project consists of an encoder-decoder architecture with skip connections. The encoder extracts features through convolutional blocks and downsampling (max-pooling) operations. The bottleneck layer further captures high-level representations. The decoder upsamples the feature maps and concatenates them with the corresponding skip connections from the encoder. The final layer predicts the probability of each pixel belonging to a tumor region using the sigmoid activation function.
The architecture of model is depicted in figure below:
![1_VUS2cCaPB45wcHHFp_fQZQ](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/e46140e0-f957-485c-933c-c15e24c58047)


**Training**

The U-Net model is trained using the Dice loss function, which is commonly used for semantic segmentation tasks. The training is performed using the AdamW optimizer with a learning rate of 0.0001. Data augmentation techniques such as random flip, rotation, and zoom are applied to increase the variability of the training data and improve the model's generalization.Model was trained on 1300 images and we got following results during training after training for 30 epochs.

![download (3)](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/30bd4d21-bcb4-4cc8-a420-07099e215089)

![download (4)](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/2c7da1d5-fddd-4ea0-8a76-f77eb4cc6e29)




**Evaluation**

The trained model is evaluated on a separate test set using various metrics such as accuracy, recall, Intersection over Union (IoU), and Dice coefficient. These metrics help assess the model's performance and effectiveness in segmenting brain tumor regions.we get following results on test data:

| Metric              | Value       |
|---------------------|-------------|
| IOU Coefficient     | 0.3043712   |
| Recall              | 0.3362936   |
| Dice Coefficient    | 0.46662897  |


After this we visualize sum of the results ,The figures given below actual mask vs predicted mask on test data:
![download](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/b1f48ed9-8f28-42aa-9ca7-2c28fa7fb13c)

![download (1)](https://github.com/SinghAnkit1010/Image-Segmentation-using-UNet/assets/103994994/4731d8d3-f6e4-4d2f-99e2-f293ab455e1d)




**Results**
The results of the model's performance and segmentation outputs on the test set will be provided in the notebook along with visualizations for further analysis.

Acknowledgments
The LGG-MRI dataset used in this project was obtained from Kaggle:(https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

The U-Net architecture implementation is based on the original paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al
