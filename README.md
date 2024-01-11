# DeepFake-Images
Implemented a Deep Learning Model to detect Real and fake Images.

This idea is implemented from a Research Paper which is provided in this Repo.

If the detected image is Fake, then it detects the source of the Fake image (General Adversive Networks (GAN) or Diffusion Models (DM)).

The model takes in the weights of a pretrained RESNET-50 model and then it fine-tunes those weights for our task by freezing the starting layers.

Level-1: Achieves 87% accuracy to detect real and Fake Images (It was trained on Dalle Fake images and Laiom Real images and tested on Glide fake images).

Level-2: The model trained for detecting the source of Fake image achieves a Testing Accuracy of 93%.

Level-3: To determine class from each gan and diffusion model , this model is still under work. (It is cuurently acheiving 55% testing accuracy, I am working on it to increase its accuracy)

# Implementation of Paper-4
I have implemented the preprocessing steps(Gaussian Blur and JPEG compression) and worked on extracting 786 dimensional feature vector from the image using ViT-L/14 pre-trained model.

The classification task using nearest neighbour and linear probing is still under work.

## Dataset Links:

Diffusion Models: [Link](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view)

GAN's: [Link](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view)
