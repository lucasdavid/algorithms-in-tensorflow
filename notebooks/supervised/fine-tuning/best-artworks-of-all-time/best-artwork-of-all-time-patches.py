#!/usr/bin/env python
# coding: utf-8

# # Supervised Fine-Tuning Best Artworks of All Time
# 
# 
# Same as the base experiment, except that it was applied to extracted patches of paintings. Sample-level classification follows the `averaging` voting strategy.
# 
# Code: [github:lucasdavid/experiments/.../supervised/fine-tuning/best-artworks-of-all-time](https://github.com/lucasdavid/experiments/blob/main/notebooks/supervised/fine-tuning/best-artworks-of-all-time/best-artworks-of-all-time.ipynb)  
# Dataset: https://www.kaggle.com/ikarus777/best-artworks-of-all-time  
# Docker image: `tensorflow/tensorflow:latest-gpu-jupyter`  

# In[1]:


from time import time
import tensorflow as tf

class RC:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    seed = 21392

class DC:
    path = '/tf/datasets/best-artworks-of-all-time'
    images = path + '/images/patches'
    info = path + '/artists.csv'

    batch_size = 32
    image_size = (299, 299)
    channels = 3
    input_shape = (batch_size, *image_size, channels)

    buffer_size = 100000

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

    class_names = np.array(sorted([item.name
                           for item in data_dir.glob('*')
                           if item.name != "LICENSE.txt"]))
    
    dataset_args = dict(
        label_mode='int',
        image_size=Config.data.image_size, batch_size=Config.data.batch_size,
        validation_split=Config.training.validation_split,
        seed=Config.run.seed)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        subset='training',
        **dataset_args)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        subset='validation',
        **dataset_args)


# ## Augmentation Policy

# In[6]:


batchwise_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom((-.3, .3)),
    tf.keras.layers.experimental.preprocessing.RandomFlip(),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
], name='batch_aug')

def augment_fn(image, label):
    image = batchwise_augmentation(image)
    image = tf.clip_by_value(image, 0, 255)
    return image, label

def prepare(ds, augment=False):
    if augment: ds = ds.map(augment_fn, num_parallel_calls=Config.run.AUTOTUNE)
    return ds.prefetch(buffer_size=Config.run.AUTOTUNE)


# In[7]:


train_ds = prepare(Data.train_ds, augment=Config.training.augment)
val_ds = prepare(Data.val_ds)
# test_ds = prepare(Data.test_ds)


# In[8]:


for x, y in train_ds:
    print('Shapes:', x.shape, 'and', y.shape)
    print("Labels: ", y.numpy())

    plt.figure(figsize=(16, 9))
    plot(x.numpy().astype(int), rows=4)
    plt.tight_layout()
    break


# ## Model Definition

# In[9]:


from tensorflow.keras.applications import inception_resnet_v2

encoder = inception_resnet_v2.InceptionResNetV2(include_top=False, pooling='avg',
                                                input_shape=Config.data.input_shape[1:])
# encoder = Model(encoder.input, encoder.get_layer('block_9_add').output)


# In[10]:


def encoder_pre(x):
    return Lambda(inception_resnet_v2.preprocess_input, name='pre_incresnet')(x)


# In[11]:


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


# In[12]:


disc.get_layer('inception_resnet_v2').trainable = False


# In[13]:


from tensorflow.keras import losses, metrics, optimizers

disc.compile(
    optimizer=optimizers.Adam(lr=Config.training.learning_rate),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        metrics.SparseCategoricalAccuracy(),
        metrics.SparseTopKCategoricalAccuracy()
    ]
)


# ## Training

# ### Initial Training for Final Classification Layer
# 
# The final layer --- currently containing random values --- must be first adjusted to match the the encoder's layers' current state.

# In[ ]:


from tensorflow.keras import callbacks

cs = [
    callbacks.TerminateOnNaN(),
    callbacks.ModelCheckpoint(Config.log.tensorboard + '/weights.h5',
                              save_best_only=True,
                              save_weights_only=True,
                              verbose=1),
    callbacks.ReduceLROnPlateau(patience=Config.training.reduce_lr_on_plateau_pacience,
                                factor=Config.training.reduce_lr_on_plateau_factor,
                                verbose=1),
    callbacks.EarlyStopping(patience=Config.training.early_stopping_patience, verbose=1),
    callbacks.TensorBoard(Config.log.tensorboard, write_graph=False)
]

try:
    disc.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.training.epochs,
        initial_epoch=0,
        callbacks=cs);
except KeyboardInterrupt:
    print('stopped')


# In[ ]:


disc.load_weights(Config.log.tensorboard + '/weights.h5')

disc.get_layer('inception_resnet_v2').trainable = True
disc.save_weights(Config.log.tensorboard + '/weights.h5')


# ### Fine-Tuning All Layers

# In[ ]:


if Config.training.epochs_fine_tuning:
    _enc = disc.get_layer('inception_resnet_v2')
    ft_layer_ix = int((1-Config.training.fine_tuning_layers)*len(_enc.layers))
    
    for ix, l in enumerate(_enc.layers):
        l.trainable = ix >= ft_layer_ix

    try: disc.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=disc.history.epoch[-1] + 1,
        epochs=len(disc.history.epoch) + Config.training.epochs_fine_tuning,
        callbacks=cs);
    except KeyboardInterrupt: print('stopped')


# In[ ]:


if Config.training.epochs_fine_tuning:
    disc.load_weights(Config.log.tensorboard + '/weights.h5')
    
    for ix, l in enumerate(_enc.layers):
        l.trainable = True
    _enc.trainable = True

    disc.save_weights(Config.log.tensorboard + '/weights.h5')


# ## Testing

# In[ ]:


disc.load_weights(Config.log.tensorboard + '/weights.h5')


# In[ ]:


from sklearn import metrics as skmetrics

def labels_and_predictions(model, ds):
    labels, predictions = [], []
    
    for x, y in ds:
        p = model(x).numpy()
        p = p.argmax(axis=1)
        
        labels.append(y.numpy())
        predictions.append(p)
    
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    labels, predictions = Data.class_names[labels], Data.class_names[predictions]
    return labels, predictions

def evaluate(model, ds):
    labels, predictions = labels_and_predictions(model, ds)
    
    print('balanced acc:', skmetrics.balanced_accuracy_score(labels, predictions))
    print('accuracy    :', skmetrics.accuracy_score(labels, predictions))
    print('Classification report:')
    print(skmetrics.classification_report(labels, predictions))


# #### Training Report

# In[ ]:


evaluate(disc, train_ds)


# #### Validation Report

# In[ ]:


evaluate(disc, val_ds)


# #### Test Report

# In[ ]:


test_ds = val_ds

evaluate(disc, test_ds)


# In[ ]:


labels, predictions = labels_and_predictions(disc, test_ds)


# In[ ]:


cm = skmetrics.confusion_matrix(labels, predictions)
cm = cm / cm.sum(axis=1, keepdims=True)
sorted_by_most_accurate = cm.diagonal().argsort()[::-1]
cm = cm[sorted_by_most_accurate][:, sorted_by_most_accurate]

plt.figure(figsize=(12, 12))
with sns.axes_style("white"):
    sns.heatmap(cm, cmap='RdPu', annot=False, cbar=False,
                yticklabels=Data.class_names[sorted_by_most_accurate], xticklabels=False);


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

plot_predictions(disc, train_ds)

