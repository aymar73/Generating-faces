# Generating-faces
Using generative adversarial networks to generate new images of faces

# Getting the project files
The project files can be found in the public GitHub repo, in the face_generation folder. You can download the files from there, but it's better to clone the repository to your computer

This way you can stay up to date with any changes made by pulling the changes to your local repository with git pull.

In order to gain more intuition about GANs in general, take a look at this blog post (https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7). Also, if you want to gain intuition about the convolution and transpose convolution arithmetic, I would suggest referring to this paper (https://arxiv.org/abs/1603.07285). For more advanced techniques on training GANs, you can refer to this paper (https://arxiv.org/abs/1606.03498).

To implement the function discriminator, the architecture is based on:
 - Use of Leaky ReLU as the activation function for the convolution layers which helps with the gradient flow. It helps alleviate the problem of sparse gradients.
 - Using same size filters across all the layers
 - Using batch normalization which stabilizes GAN training.
 - Using Sigmoid as the activation function for the output layer which produces probability-like values between 0 and 1
We could improve the discriminator function further for better results by:
 - Adding more convolution transpose layers. Using a simple 3-layer generator is not going to help generate better images. Apart from this, generators which are sufficiently larger than the discriminator in terms of depth, help generate better samples. So, I will add at least 2 more layers to the generator.
 - Using a stride of 1 in the last layer to avoid checkerboard-like artifacts in the generated images.We can read more about this phenomenon in this blog post (https://distill.pub/2016/deconv-checkerboard/).
 - Using a smaller dropout rate for generator as compared to the discriminator.
 - Using a bigger value, between 0.1 or 0.2, for leak parameter. 0.01 is too small.
 
I used control dependency on update ops before optimization. Normally, we scale the inputs to a neural network between 0 and 1. However, as the data moves through multiple layers, it starts shifting from that distribution and the deeper layers start getting data in a different range as input. This is known as internal covariate shift (https://arxiv.org/abs/1502.03167). To combat this, a technique known as Batch normalization was introduced, where the inputs to the layer are scaled between 0 and 1. The moving mean and variance need to be updated for the technique. In Tensorflow, these operations are part of update ops and these need to be updated prior to the optimization. So, we add control dependencies on the update ops before optimizing the network. Refer to this post for more information (http://ruishu.io/2016/12/27/batchnorm/).
