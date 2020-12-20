# Algorithms in Tensorflow

Experiments with tensorflow 2.0, gpu support,
persistent logging and stable docker env.

## Usage
```shell
./actions/run.sh {up,down,build}
```

## Summary

* Supervised
  * classification
    * [churn](notebooks/supervised/classification/churn)
    * [facts](notebooks/supervised/classification/facts)
  * fine-tuning
    * [cifar10](notebooks/supervised/fine-tuning/cifar10)
    * [best-artworks-of-all-time](notebooks/supervised/fine-tuning/best-artworks-of-all-time)
  * segmentation
    * [pix2pix](notebooks/supervised/segmentation/pix2pix.ipynb)
* Unsupervised
  * style-transfer
    * [gatys](notebooks/unsupervised/style-transfer/gatys.ipynb), Image Style Transfer Using Convolutional Neural Networks [article](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
  * variational-autoencoder
    * [vae](notebooks/unsupervised/variational-autoencoder/vae.ipynb), Tutorial on Variational Autoencoders [article](https://arxiv.org/pdf/1606.05908.pdf)
    * [vae-cifar10](notebooks/unsupervised/variational-autoencoder/vae-cifar10.ipynb)
    * [vae-fashion-mnist](notebooks/unsupervised/variational-autoencoder/vae-fashion-mnist.ipynb)
  * contrastive
    * [contrastive-loss](notebooks/unsupervised/contrastive/contrastive-loss.ipynb), Understanding the Behavior of Contrastive Loss [article](https://arxiv.org/pdf/2012.09740.pdf)
* Explaining
  * CAM Methods
    * [activation-maximization](notebooks/explaining/activation-maximization.ipynb)
    * [Grad CAM](notebooks/explaining/cam-gradcam.ipynb)
    * [Grad CAM++](notebooks/explaining/cam-gradcam++.ipynb)
    * [Score CAM](notebooks/explaining/cam-gradcam-score.ipynb)
  * Gradient-based Saliency Methods
    * [Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation.ipynb)
    * [Smooth Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-smooth.ipynb)
    * [Guided Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-guided.ipynb)
    * [FullGradients Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-full.ipynb)
