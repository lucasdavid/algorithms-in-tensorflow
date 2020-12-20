#!/usr/bin/env python
# coding: utf-8

# # Check Results of Any Given Model
# 
# Code: [github:lucasdavid/experiments/.../supervised/fine-tuning/best-artworks-of-all-time/results](https://github.com/lucasdavid/experiments/blob/main/notebooks/supervised/fine-tuning/best-artworks-of-all-time/results.ipynb)  
# Dataset: https://www.kaggle.com/ikarus777/best-artworks-of-all-time  
# Docker image: `tensorflow/tensorflow:latest-gpu-jupyter`  

# In[1]:


from time import time
import tensorflow as tf

class RC:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    base_path = ('/tf/logs/baoat/d:baoat-patches e:200 fte:0 b:32 v:0.3 '
                 'm:inceptionv3 aug:True/1610567121')
    model = f'{base_path}/weights.h5'
    results = f'{base_path}/eval'
    seed = 1402

class DC:
    path = '/tf/datasets/best-artworks-of-all-time'
    images = path + '/images/patches'
    info = path + '/artists.csv'

    batch_size = 32
    image_size = (299, 299)
    channels = 3
    input_shape = (batch_size, *image_size, channels)

    buffer_size = 100000

    batches_evaluated = 512  # 1024

class TC:
    epochs = 200
    learning_rate = .001
    validation_split = .3
    reduce_lr_on_plateau_pacience = 20
    reduce_lr_on_plateau_factor = .5
    
    early_stopping_patience = 50
    
    splits = [f'train[{validation_split}:]', f'train[:{validation_split}]', 'test']
    
    augment = True
    
    epochs_fine_tuning = 0
    learning_rate_fine_tuning = .0005
    fine_tuning_layers = .2  # 20%
    
class LogConfig:
    tensorboard = (f'/tf/logs/d:baoat-patches '
                   f'e:{TC.epochs} fte:{TC.epochs_fine_tuning} b:{DC.batch_size} '
                   f'v:{TC.validation_split} m:inceptionv3 aug:{TC.augment}'
                   f'/{int(time())}')
    
class Config:
    run = RC
    data = DC
    training = TC
    log = LogConfig


# ## Setup

# In[2]:


import os
import pathlib
from math import ceil

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, BatchNormalization,
                                     Activation, Lambda)


# In[3]:


def plot(y, titles=None, rows=1, i0=0):
    for i, image in enumerate(y):
        if image is None:
            plt.subplot(rows, ceil(len(y) / rows), i0+i+1)
            plt.axis('off')
            continue

        t = titles[i] if titles else None
        plt.subplot(rows, ceil(len(y) / rows), i0+i+1, title=t)
        plt.imshow(image)
        plt.axis('off')


# In[4]:


sns.set()


# ## Dataset

# In[5]:

class Data:
    info = pd.read_csv(Config.data.info)
    data_dir = pathlib.Path(Config.data.images)

    dataset_args = dict(
        label_mode='int',
        image_size=Config.data.image_size, batch_size=Config.data.batch_size,
        validation_split=Config.training.validation_split,
        shuffle=True,
        seed=Config.run.seed)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        subset='training',
        **dataset_args)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        subset='validation',
        **dataset_args)
    
    class_names = np.asarray(train_ds.class_names)


# In[6]:


def prepare(ds, augment=False):
    return ds.cache().prefetch(buffer_size=Config.run.AUTOTUNE)

train_ds = prepare(Data.train_ds, augment=Config.training.augment)
val_ds = prepare(Data.val_ds)
test_ds = val_ds  # prepare(Data.test_ds)


# In[7]:


for x, y in train_ds:
    print('Shapes:', x.shape, 'and', y.shape)
    print("Labels: ", y.numpy())

    plt.figure(figsize=(16, 9))
    plot(x.numpy().astype(int), rows=4)
    plt.tight_layout()
    break


# ## Model Definition

# In[8]:


from tensorflow.keras.applications import inception_resnet_v2

encoder = inception_resnet_v2.InceptionResNetV2(include_top=False, pooling='avg',
                                                input_shape=Config.data.input_shape[1:])
# encoder = Model(encoder.input, encoder.get_layer('block_9_add').output)


# In[9]:


def encoder_pre(x):
    return Lambda(inception_resnet_v2.preprocess_input, name='pre_incresnet')(x)


# In[10]:


from tensorflow.keras.layers import GlobalAveragePooling2D

def dense_block(x, units, activation='relu', name=None):
    y = Dense(units, name=f'{name}_fc', use_bias=False)(x)
    y = BatchNormalization(name=f'{name}_bn')(y)
    y = Activation(activation, name=f'{name}_relu')(y)
    return y
    
def discriminator():
    y = x = Input(shape=Config.data.input_shape[1:], name='inputs')
    y = encoder_pre(y)
    y = encoder(y)
    y = Dense(len(Data.class_names), name='predictions')(y)
    return tf.keras.Model(x, y, name='author_disc')

disc = discriminator()
disc.summary()


# In[11]:


from tensorflow.keras import losses, metrics, optimizers

disc.compile(
    optimizer=optimizers.Adam(lr=Config.training.learning_rate),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        metrics.SparseCategoricalAccuracy(),
        metrics.SparseTopKCategoricalAccuracy()
    ]
)


# ## Testing

# In[12]:


disc.load_weights(Config.run.model)


# In[20]:


def labels_and_predictions(model, ds):
    labels, predictions = [], []
    
    for x, y in ds:
        p = model.predict(x).argmax(axis=1)
        y = y.numpy()

        labels.append(y)
        predictions.append(p)
        
    return np.concatenate(labels), np.concatenate(predictions)


# In[ ]:


from sklearn import metrics as skmetrics


def painting_from(patch_name):
    n = os.path.basename(patch_name)
    n, _ = os.path.splitext(n)
    *n, ix = n.split('_')
    return '_'.join(n), int(ix)


def evaluate(labels, predictions, names):
    print('Sample-level Report')
    print('  balanced acc         :', skmetrics.balanced_accuracy_score(labels, predictions))
    print('  accuracy             :', skmetrics.accuracy_score(labels, predictions))
    print('  classification report:')
    print(skmetrics.classification_report(labels, predictions))
    
    n = names[:Config.data.batches_evaluated * Config.data.batch_size]
    n, idxs = zip(*(painting_from(n) for n in n))

    results = pd.DataFrame({'label': l, 'pred':p, 'name': n, 'indices': idxs})
    return results


# #### Test Report

# In[ ]:


sample_ds = test_ds.take(Config.data.batches_evaluated)
l, p = labels_and_predictions(disc, sample_ds)
names = Data.val_ds.file_paths

results = evaluate(l, p, names)
results.to_csv(Config.run.results + '/predictions.csv', index=False)


# In[ ]:


cm = skmetrics.confusion_matrix(l, p)
cm = cm / cm.sum(axis=1, keepdims=True)
sorted_by_most_accurate = cm.diagonal().argsort()[::-1]
cm = cm[sorted_by_most_accurate][:, sorted_by_most_accurate]

plt.figure(figsize=(12, 12))
with sns.axes_style("white"):
    sns.heatmap(cm, cmap='RdPu', annot=False, cbar=False,
                yticklabels=Data.class_names[sorted_by_most_accurate], xticklabels=False);

plt.savefig(Config.run.results + '/test-cm.jpg')


# In[ ]:


def plot_predictions(model, ds, take=1):
    figs, titles = [], []
    
    plt.figure(figsize=(16, 16))
    for ix, (x, y) in enumerate(ds.take(take)):
        p = model.predict(x)
        p = tf.nn.softmax(p).numpy()
        figs.append(x.numpy().astype(int))
        titles.append([f'label: {a}\npredicted: {b}\nproba:{c:.0%}'
                       for a, b, c in zip(Data.class_names[y],
                                          Data.class_names[p.argmax(axis=-1)],
                                          p.max(axis=-1))])
    plot(np.concatenate(figs),
         titles=sum(titles, []),
         rows=6)
    plt.tight_layout()

plot_predictions(disc, test_ds.take(1))
plt.savefig(Config.run.results + '/test-predictions.jpg')
