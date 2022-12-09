#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sys


data_loc = sys.argv[1]
# basic datasets: train and val at 64x64
train_img_small = np.load(data_loc+'/training/train_img_64.npy')
train_img_small = np.float32(train_img_small) / 255

#train_geom = np.concatenate([np.load("datasets/train_geom_img.npy"), np.load("datasets/train_geom_wrl.npy")], axis=1)
train_geom = np.concatenate([
            np.load(data_loc+'/training/train_geom_img.npy'),
            np.load(data_loc+'/training/train_geom_wrl.npy')
            ],
            axis=1)
train_geom = train_geom.reshape((-1, 21*6))

train_lbl = np.load(f'{data_loc}/training/train_lbl.npy')
print(train_img_small.shape, train_geom.shape, train_lbl.shape)


val_img_small = np.load(f'{data_loc}/validation/val_img_64.npy')
val_img_small = np.float32(val_img_small) / 255

val_geom = np.concatenate([
    np.load(f'{data_loc}/validation/val_geom_img.npy'),
    np.load(f'{data_loc}/validation/val_geom_wrl.npy')
    ],
    axis=1)
val_geom = val_geom.reshape((-1,21*6))

val_lbl = np.load(f'{data_loc}/validation/val_lbl.npy')
print(val_img_small.shape, val_geom.shape, val_lbl.shape)


# synthetic data pre-generated by albumentations
train_img_synth = np.concatenate([
    np.load(f'{data_loc}/synthetic/train_img_64_synth.npy'),
    np.load(f'{data_loc}/training/train_img_64.npy')
    ],
    axis=0)

train_lbl_synth = np.concatenate([
    np.load(f'{data_loc}/synthetic/train_lbl_64_synth.npy'),
    np.load(f'{data_loc}/training/train_lbl.npy')
    ],
    axis=0)

train_geom_synth = np.concatenate([
    np.concatenate([
        np.load(f'{data_loc}/synthetic/train_geom_img_64_synth.npy'),
        np.load(f'{data_loc}/synthetic/train_geom_wrl_64_synth.npy')
        ],
        axis=1),
    np.concatenate([        
        np.load(f'{data_loc}/training/train_geom_img.npy'),
        np.load(f'{data_loc}/training/train_geom_wrl.npy'),
        ],
        axis=1)
    ],
    axis=0)

train_img_synth = np.float32(train_img_synth) / 255
train_geom_synth = train_geom_synth.reshape((-1, 21*6))
print(train_img_synth.shape, train_geom_synth.shape, train_lbl_synth.shape)

synthesiser_train = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.2, value_range=(0,1)),
                                    tf.keras.layers.RandomFlip(mode = 'horizontal'),
                                    tf.keras.layers.RandomRotation(0.05, fill_mode='constant'),
                                    tf.keras.layers.RandomZoom(height_factor=0.2, fill_mode='constant')])
batch_size = train_img_synth.shape[0]
train_img_tf = tf.data.Dataset.from_tensor_slices(train_img_synth)
train_geom_tf = tf.data.Dataset.from_tensor_slices(train_geom_synth).batch(batch_size).get_single_element()


# this makes a dataset object with an attached function, rather than just applying a function once to its tensors
train_synth = train_img_tf.map(lambda x: synthesiser_train(x),
                                 num_parallel_calls=batch_size).batch(batch_size)
train_proc = train_synth.get_single_element()

# had to add @tf.function to make models saveable
class testAttentionModel(tf.keras.Model):
    def __init__(self, conv_filters, reg_coef=0, labels=50, use_geom_backup=True):
        super(testAttentionModel, self).__init__()
        filters_1, filters_2, filters_3 = conv_filters
        conv_out_size = filters_3
        self.reg = tf.keras.regularizers.L2(reg_coef)
        self.spatial_dropout_prob = 0.0
        self.dropout_prob = 0.1
        self.use_geom_backup = use_geom_backup
        
        
        # 64x64xch
        self.conv_1a = tf.keras.layers.Convolution2D(filters_1, 5, padding='same', use_bias=True, 
                                                     activation='relu',
                                                     kernel_regularizer=self.reg)
        self.conv_1b = tf.keras.layers.Convolution2D(filters_1, 3, padding='same', use_bias=True, 
                                                     activation='relu',
                                                     kernel_regularizer=self.reg)
        
        self.batch1 = tf.keras.layers.BatchNormalization()
        
        # 32x32xch
        self.conv_2a = tf.keras.layers.Convolution2D(filters_2, 3, padding='same', use_bias=True, 
                                                     activation='relu',
                                                     kernel_regularizer=self.reg)
        self.conv_2b = tf.keras.layers.Convolution2D(filters_2, 3, padding='same', use_bias=True,
                                                     activation='relu',
                                                     kernel_regularizer=self.reg)
        # 16x16xch
        self.conv_3a = tf.keras.layers.Convolution2D(filters_3, 3, padding='same', use_bias=True, 
                                                     activation='relu',
                                                    kernel_regularizer=self.reg)
        self.conv_3b = tf.keras.layers.Convolution2D(filters_3, 3, padding='same', use_bias=True, 
                                                     activation='relu',
                                                     kernel_regularizer=self.reg)
        # 8x8xch
        #self.conv_4a = tf.keras.layers.Convolution2D(filters_4, 3, padding='same', use_bias=True, activation='relu')
                                                    # activation='tanh', kernel_regularizer=regulator)
        #self.conv_4b = tf.keras.layers.Convolution2D(filters_4, 3, padding='same', use_bias=True, activation='relu')
                                                     #activation='tanh', kernel_regularizer=regulator)
        # out: 4x4xch
        
        self.geom1 = tf.keras.layers.Dense(64, use_bias=True, activation='relu', kernel_regularizer=self.reg)
        self.geom_backup = tf.keras.layers.Dense(64, use_bias=True, activation='relu', kernel_regularizer=self.reg)
        self.geom2 = tf.keras.layers.Dense(64, use_bias=True, activation='relu', kernel_regularizer=self.reg)
        self.attention = tf.keras.layers.Dense(conv_out_size, use_bias=True, 
                                               activation='softmax', kernel_regularizer=self.reg)

        self.classifier = tf.keras.layers.Dense(labels, activation='softmax')
        
    @tf.function
    def call(self, input_list, training=True):
        c_out = tf.keras.layers.GaussianNoise(0.03)(input_list[0], #start 64x64x3
                                                    training=training)
        c_out = tf.keras.layers.SpatialDropout2D(self.spatial_dropout_prob)(self.conv_1a(c_out),
                                                                            training=training)
        
        c_out = self.batch1(c_out)
        
        c_out = tf.keras.layers.MaxPool2D(strides=(2,2))(self.conv_1b(c_out)) # to 32x32xch
        c_out = tf.keras.layers.SpatialDropout2D(self.spatial_dropout_prob)(self.conv_2a(c_out),
                                                                            training=training)
        
        c_out = self.batch1(c_out)
        
        c_out = tf.keras.layers.MaxPool2D(strides=(2,2))(self.conv_2b(c_out))
        c_out = tf.keras.layers.SpatialDropout2D(self.spatial_dropout_prob) (self.conv_3a(c_out),
                                                                            training=training)
        
        c_out = self.batch1(c_out)
        
        c_out = tf.keras.layers.MaxPool2D(strides=(2,2))(self.conv_3b(c_out)) # to 16x16xch
        #c_out = self.conv_4a(c_out)
        #c_out = tf.keras.layers.MaxPool2D(strides=(2,2))(self.conv_4b(c_out))
       
        if tf.math.reduce_max(input_list[1]) == 0 and self.use_geom_backup:
            g_out = self.geom_backup(tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling2D(pool_size=(4, 4),
                                                                                                strides=(4,4),
                                                                                                padding='valid')(input_list[0])))
        else:
            g_out = self.geom1(input_list[1]) # if use_geom_backup is off this will output max(0, bias)
        g_out = tf.keras.layers.Dropout(self.dropout_prob)(g_out, training=training)
        g_out = tf.keras.layers.Dropout(self.dropout_prob)(self.geom2(g_out),training=training)
        g_out = tf.keras.layers.Dropout(self.dropout_prob)(self.attention(g_out),training=training)
        g_out = tf.expand_dims(tf.expand_dims(g_out, axis=-2), axis=-2)
       
        return self.classifier(tf.keras.layers.Flatten()(tf.math.multiply(c_out, g_out)))
    
testAttender = testAttentionModel((64,128,256), reg_coef=0.0001)

learning_rate=0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    learning_rate,
                    decay_steps=5000,
                    decay_rate=0.9,
                    staircase=True)

testAttender.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
NUM_EPOCHS = 40
history = testAttender.fit([train_proc, train_geom_tf], train_lbl_synth,
                 validation_data=([val_img_small, val_geom], val_lbl),
                 epochs=NUM_EPOCHS)


# ## Saving the model ##
save_loc = sys.argv[2]

# This code cell saves the model's architecture 
# to a folder name save_loc + 'static_80_ep'
# see https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format
testAttender.save(f'{save_loc}/static_40_ep')


# ## Plotting ##
# This code cell plots the training and val accuracy by epoch
history_dict = history.history
sns.set_style("whitegrid")

plt.figure(figsize=(10,6))

plt.scatter(range(1,NUM_EPOCHS+1), 
            history_dict['accuracy'], 
            label="Training Data")
plt.scatter(range(1,NUM_EPOCHS+1), 
            history_dict['val_accuracy'], 
            marker='v',
            label="Validation Data")

plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)
print(f"saving to {save_loc}/train_val_accuracy_by_epoch.png")
plt.savefig(f'{save_loc}/train_val_accuracy_by_epoch.png', dpi=100)