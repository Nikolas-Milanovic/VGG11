# VGG11
Implemented the VGG11 Convolutional Neural Network (CNN) architecture as cited in model A in Table 1 of this [academic paper](https://arxiv.org/pdf/1409.1556)

## Results: 95%+ Test Accuracy
* 95% was achieved with only 5 epochs
* Tested and Trained using the [MNIST Dataset](https://pytorch.org/vision/stable/datasets.html#mnist)
* Batch normalization was applied to improve convergence rate
* Future optimization could include learning rate scheduling 

## Plots (Test & Training Accuracy, Test & Training Loss vs Epochs):

<img width="181" alt="image" src="https://github.com/Nikolas-Milanovic/VGG11/assets/59632554/6f838d86-7cd9-4fba-88a2-c1390ee6483d">

<img width="183" alt="image" src="https://github.com/Nikolas-Milanovic/VGG11/assets/59632554/e64e8866-559f-4e73-b779-093a5710d5f4">

<img width="188" alt="image" src="https://github.com/Nikolas-Milanovic/VGG11/assets/59632554/d75fd3c7-51db-4a15-b205-eb906f3859eb">

<img width="194" alt="image" src="https://github.com/Nikolas-Milanovic/VGG11/assets/59632554/3eaf1d4c-5280-49ba-ae34-4cb889b026f6">

## Model Generalization
* Applied regularization techniques such as data augmentation
* Data augmentation was applied by randomly applying transformations to the dataset by either flipping horizontally, veritcally, or applying an image blur (Gaussian noise, var = [0.01, 0.1, 1]). As seen below: 
<img width="611" alt="image" src="https://github.com/Nikolas-Milanovic/VGG11/assets/59632554/7c09a23a-f8be-4d2e-b2de-21557c4229d8">

  





