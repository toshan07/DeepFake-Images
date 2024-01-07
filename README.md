# DeepFake-Images
Implemented a Deep Learning Model to detect Real and fake Images.

This idea is implemented from a Research Paper which is provided in this Repo.

If the detected image is Fake, then it detects the source of the Fake image (General Adversive Networks (GAN) or Diffusion Models (DM)).

The model takes in the weights of a pretrained RESNET-50 model and then it fine-tunes those weights for our task.

The above-mentioned task achieves 87% accuracy to detect real and Fake Images (It was trained on Dalle Fake images and Laiom Real images and tested on Glide fake images).

The model trained for detecting the source of Fake image achieves a Testing Accuracy of 93%.

## Dataset Links:

Diffusion Models: [URL link](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view)

GAN's: [URL Link](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view)
