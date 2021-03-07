# Algorithms in Tensorflow

Experiments with tensorflow 2.0, gpu support,
persistent logging and stable docker env.

## Usage
```shell
./actions/run.sh {up,down,build}
```

## Summary

| Section | Kind | Description | References |
| --- | --- | --- | --- |
| [churn](notebooks/supervised/classification/churn) | supervised classification | Study over client churn behavior | |
| [facts](notebooks/supervised/classification/facts) | supervised classification | Study over a set of true and fake news | |
| [cifar10](notebooks/supervised/fine-tuning/cifar10) | fine-tuning classification | Cifar10 | |
| [best-artworks-of-all-time](notebooks/supervised/fine-tuning/best-artworks-of-all-time) | fine-tuning | Art authorship attribution | |
| [pix2pix](notebooks/supervised/segmentation/pix2pix.ipynb) | supervised segmentation | | |
| [gatys](notebooks/unsupervised/style-transfer/gatys.ipynb) | unsupervised | style transfer | [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) |
| [Variational Autoencoder](notebooks/unsupervised/variational-autoencoder/vae.ipynb) | unsupervised | vae | [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf) |
| [VaE Cifar10](notebooks/unsupervised/variational-autoencoder/vae-cifar10.ipynb) | unsupervised | vae | [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf) |
| [VaE Fashion MNIST](notebooks/unsupervised/variational-autoencoder/vae-fashion-mnist.ipynb) | unsupervised | vae | [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf) |
| [Contrastive loss](notebooks/unsupervised/contrastive/contrastive-loss.ipynb) | unsupervised | contrastive siamese discriminator | [Understanding the Behavior of Contrastive Loss](https://arxiv.org/pdf/2012.09740.pdf) |
| [activation-maximization](notebooks/explaining/activation-maximization.ipynb) | Attention | CAM | |
| [Grad CAM](notebooks/explaining/cam-gradcam.ipynb) | Attention | CAM | |
| [Grad CAM++](notebooks/explaining/cam-gradcam++.ipynb) | Attention | CAM | |
| [Score CAM](notebooks/explaining/cam-gradcam-score.ipynb) | Attention | CAM | |
| [Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation.ipynb) | Saliency | Gradient-based | |
| [Smooth Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-smooth.ipynb) | Saliency | Gradient-based | |
| [Guided Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-guided.ipynb) | Saliency | Gradient-based | |
| [FullGradients Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-full.ipynb) | Saliency | Gradient-based | |
