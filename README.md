# [![Tensorflow Logo](assets/tf.png)](https://tensorflow.org) Algorithms in Tensorflow

Experiments with tensorflow 2.0, gpu support, persistent logging and stable docker env.

## Summary
### Supervised
| Notebook | Kind | Description | References |
| --- | --- | --- | --- |
| [Intro](notebooks/supervised/classification/introduction.ipynb) | classification | Introduction over classification, using Logistic Regressison and training with Grid Search Cross-validation | [dataset](https://osf.io/ja9dw/) |
| [Multi-class](notebooks/supervised/classification/multiclass.ipynb) | classification | Introduction to multi-class classification and examples | |
| [Study of Churn](notebooks/supervised/classification/churn) | classification | Study over client churn behavior | |
| [Study of Facts](notebooks/supervised/classification/facts) | classification | Study over a set of true and fake news | |
| [Churn](notebooks/supervised/regression/intro.ipynb) | regression | Introduction over linear regression | |
| [Study of Weather WW2](notebooks/supervised/regression/weather.ipynb) | regression | Instantial study over Weather information during WW2 | [dataset](https://www.kaggle.com/smid80/weatherww2/data) |
| [cifar10](notebooks/supervised/fine-tuning/cifar10) | fine-tuning  classification | CNN fine-tuned from ImageNet to solve Cifar10 | [dataset](https://www.cs.toronto.edu/~kriz/cifar.html) |
| [best-artworks-of-all-time](notebooks/supervised/fine-tuning/best-artworks-of-all-time) | fine-tuning classification | Art authorship attribution fine-tuned from ImageNet | [BAoAT dataset](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) |
| [Study of Mapping Challenge](notebooks/supervised/segmentation/unet/mapping-challenge-efficientnetb4.ipynb) | semantic segmentation | Segmentation of construction satellite images using U-NET and EfficientNetB4 | [Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)  |
| [Study of Oxford IIT Pet](notebooks/supervised/segmentation/unet/oxford-iit-pet-mobilenetv2.ipynb) | semantic segmentation | Segmentation of dogs and cats from the `oxford_iit_pet` | [Oxford IIT Pet dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) |

### UnsupervisedGradient-based
| Notebook | Kind | Description | References |
| --- | --- | --- | --- |
| [gatys](notebooks/unsupervised/style-transfer/gatys.ipynb) | style transfer | Style transfer between two arbitrary images | [article](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) |
| [VaE Cifar10](notebooks/unsupervised/variational-autoencoder/vae-cifar10.ipynb) | variational autoencoder | VaE over Cifar10 and embedding of the set in the metric space | [tutorial](https://arxiv.org/pdf/1606.05908.pdf) |
| [VaE Fashion MNIST](notebooks/unsupervised/variational-autoencoder/vae-fashion-mnist.ipynb) | variational autoencoder | VaE over Fashion MNIST | [tutorial](https://arxiv.org/pdf/1606.05908.pdf) |
| [Contrastive loss](notebooks/unsupervised/contrastive/contrastive-loss.ipynb) | siamese network | Siamese CNN trained with contrastive loss | [article](https://arxiv.org/pdf/2012.09740.pdf) |

### Explaining
| Notebook | Kind | Description | References |
| --- | --- | --- | --- |
| [Activation Maximization](notebooks/explaining/activation-maximization.ipynb) | optimization | Network concept-representation by gradient ascending over the output value | [class notes](https://slazebni.cs.illinois.edu/fall18/lec11_visualization.pdf) |
| [Grad-CAM](notebooks/explaining/cam-gradcam.ipynb) | CAM | Explaining networks' decision using gradient info and CAM | [article](https://arxiv.org/abs/1610.02391) |
| [Grad-CAM++](notebooks/explaining/cam-gradcam++.ipynb) | CAM | Adjust Grad-CAM weights to prevent activation domiance of large regions over small ones | [article](https://arxiv.org/abs/1710.11063) |
| [Score-CAM](notebooks/explaining/cam-gradcam-score.ipynb) | CAM | CAM based on Increase of Confidence | [article](https://arxiv.org/abs/1910.01279) |
| [Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation.ipynb) | saliency | Gradient-based | |
| [Guided Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-guided.ipynb) | saliency | Gradient-based explaining method considering positive intermediate gradients | [article](https://arxiv.org/pdf/1412.6806.pdf)  |
| [Smooth Gradient Backpropagation](notebooks/explaining/saliency-gradient-backpropagation-smooth.ipynb) | saliency | Gradient-based explaining method with local-level gradient correction | [article](https://arxiv.org/pdf/1706.03825.pdf) |
| [Full Gradient Representation](notebooks/explaining/saliency-gradient-backpropagation-full.ipynb) | saliency | Explaining using function linearization with gradient-based and bias information | [article](https://arxiv.org/pdf/1905.00780.pdf) |


## Usage
Code in this repository is kept inside jupyter notebooks, so any jupyter
server will do. I added a docker-compose env to simplify things, which can
be used as follows:
```shell
./actions/run.sh                                          # start jupyter notebook
./actions.run.sh {up,down,build}                          # more compose commands
./actions.run.sh exec experiments python path/to/file.py  # any commands, really
./actions/run.sh tensorboard                              # start tensorboard
```
