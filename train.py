from keras.layers import Dense, Dropout, Activation, Input,Lambda
#from keras.layers import Convolution2D, MaxPooling2D, Input
#from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, SGD
from keras import backend as K
import os
import numpy as np
import scipy.misc
from PIL import Image

 
batch_size = 60


def quadruple_loss(X):
    anchor_image, pos_text, neg_text, anchor_text, pos_image, neg_image = X

    anchor_image = K.l2_normalize(anchor_image, axis=-1)
    pos_text = K.l2_normalize(pos_text, axis=-1)
    neg_text = K.l2_normalize(neg_text, axis=-1)
    anchor_text = K.l2_normalize(anchor_text, axis=-1)
    pos_image = K.l2_normalize(pos_image, axis=-1)
    neg_image = K.l2_normalize(neg_image, axis=-1)

    anchor_pos_dist = K.sum(K.square(anchor_image - pos_text), axis=-1)
    anchor_neg_dist = K.sum(K.square(anchor_image - neg_text), axis=-1)
    loss1 = K.maximum(0.0, 0.5 + anchor_pos_dist - anchor_neg_dist)

    anchor_pos_dist = K.sum(K.square(anchor_text - pos_image), axis=-1)
    anchor_neg_dist = K.sum(K.square(anchor_text - neg_image), axis=-1)
    loss2 = K.maximum(0.0, 0.5 + anchor_pos_dist - anchor_neg_dist)
    loss = 0.3*loss1 + 0.7*loss2
    return loss


def identify_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true) 


def create_image_network(input_dim):
        model = Sequential()
        model.add(Dense(64, input_shape=input_dim, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(16, activation='relu'))
        #model.add(BatchNormalization())
        return model

def create_text_network(input_dim):
        model = Sequential()
        model.add(Dense(64, input_shape=input_dim, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(16, activation='relu'))
        #model.add(BatchNormalization())
        return model

# create image base network
input_image_dim = (128,)
base_image_network = create_image_network(input_image_dim)

input_anchor_image = Input(shape=(input_image_dim))
anchor_image = base_image_network(input_anchor_image)

input_pos_image = Input(shape=(input_image_dim))
pos_image = base_image_network(input_pos_image)

input_neg_image = Input(shape=(input_image_dim))
neg_image = base_image_network(input_neg_image)

# create text base network
input_text_dim = (147,)
base_text_network = create_text_network(input_text_dim)

input_anchor_text = Input(shape=(input_text_dim))
anchor_text = base_text_network(input_anchor_text)

input_pos_text = Input(shape=(input_text_dim))
pos_text = base_text_network(input_pos_text)

input_neg_text = Input(shape=(input_text_dim))
neg_text = base_text_network(input_neg_text)

# merge four loss
#secLoss = merge([anchor_image, pos_text, neg_text, anchor_text, pos_image, neg_image],
#                mode=quadruple_loss,
#                name='quad',
#                output_shape=(1, ))

secLoss = Lambda(quadruple_loss,
                  name='quad',
                  output_shape=(1, ))([anchor_image, pos_text, neg_text, anchor_text, pos_image, neg_image])


model = Model(inputs=[input_anchor_image, input_pos_text, input_neg_text, input_anchor_text, input_pos_image, input_neg_image], outputs=[secLoss])

#optimizer = RMSprop()
optimizer = SGD(lr = 0.01, momentum = 0.98, decay = 0.0, nesterov = True)
model.compile(optimizer = optimizer, loss={'quad': identify_loss})

def common_generator(_idxs, image_feats, text_feats, i):
        anchor_image = np.zeros((batch_size, 128), dtype="float32")
        pos_image = np.zeros((batch_size, 128), dtype="float32")
        neg_image = np.zeros((batch_size, 128), dtype="float32")
        anchor_text = np.zeros((batch_size, 147), dtype="float32")
        pos_text = np.zeros((batch_size, 147), dtype="float32")
        neg_text = np.zeros((batch_size, 147), dtype="float32")
        for k in range(batch_size):
                # anchor point
                anchor_idx = int(_idxs[i * batch_size + k, 0])
                anchor_image[k, :] = image_feats[anchor_idx, :]
                anchor_text[k, :] = text_feats[anchor_idx, :]

                # positive point
                pos_idx = int(_idxs[i * batch_size + k, 1])
                pos_image[k, :] = image_feats[pos_idx, :]
                pos_text[k, :] = text_feats[pos_idx, :]

                # negative point
                neg_idx = int(_idxs[i * batch_size + k, 2])
                neg_image[k, :] = image_feats[neg_idx, :]
                neg_text[k, :] = text_feats[neg_idx, :]
        return (anchor_image, pos_text, neg_text, anchor_text, pos_image, neg_image, np.ones((batch_size,1)))


def train_generator(): 
    while True:
        for i in range(chic_train_num/batch_size):
            # read images
            (a, b, c, d, e, f, h) = common_generator(chic_train_idx, chic_image_train, chic_text_train, i)
            yield [a, b, c, d, e, f], [h]
        for j in range(radar_train_num/batch_size):
            # read images
            (a, b, c, d, e, f, h) = common_generator(radar_train_idx, radar_image_train, radar_text_train, j)
            yield [a, b, c, d, e, f], [h]
        for k in range(tradesy_train_num/batch_size):
            # read images
            (a, b, c, d, e, f, h) = common_generator(tradesy_train_idx, tradesy_image_train, tradesy_text_train,  k)
            yield [a, b, c, d, e,f], [h]

def valid_generator(): 
    while True:
        for i in range(chic_val_num/batch_size):
            # read images
            (a, b, c, d,  e, f, h) = common_generator(chic_validation_idx, chic_image_validation, chic_text_validation, i)
            yield [a, b, c, d, e, f], [h]
        for j in range(radar_val_num/batch_size):
            # read images
            (a, b, c, d, e, f, h) = common_generator(radar_validation_idx, radar_image_validation, radar_text_validation, j)
            yield [a, b, c, d, e, f], [h]
        for k in range(tradesy_val_num/batch_size):
            # read images
            (a, b, c, d, e, f, h) = common_generator(tradesy_validation_idx, tradesy_image_validation, tradesy_text_validation, k)
            yield [a, b, c, d, e, f], [h]

source_path = '/home/belizabeth/Documents/Programming/Keras/TMM_2017/'
data_path = source_path + 'sources/train_model/extractor/quintuple/feats_npy_times_3/'

chic_train_idx = np.loadtxt(source_path + 'sources/sampling_triplets/chic/chic_train_sample.txt')
chic_image_train = np.load(data_path + 'chic_train.npy')
chic_text_train = np.loadtxt(source_path + 'labels/chic/chic_train_full_labels.txt')
chic_train_num = chic_train_idx.shape[0]
print chic_train_idx.shape[0]
print chic_image_train.shape[0]
print chic_text_train.shape[0]

chic_validation_idx = np.loadtxt(source_path + 'sources/sampling_triplets/chic/chic_validation_sample.txt')
chic_image_validation = np.load(data_path + 'chic_validation.npy')
chic_text_validation = np.loadtxt(source_path + 'labels/chic/chic_validation_full_labels.txt')
chic_val_num = chic_validation_idx.shape[0]
print chic_validation_idx.shape[0]
print chic_image_validation.shape[0]
print chic_text_validation.shape[0]

radar_train_idx = np.loadtxt(source_path + 'sources/sampling_triplets/radar/radar_train_sample.txt')
radar_image_train = np.load(data_path + 'radar_train.npy')
radar_text_train = np.loadtxt(source_path + 'labels/radar/radar_train_full_labels.txt')
radar_train_num = radar_train_idx.shape[0]
print radar_train_idx.shape[0]
print radar_image_train.shape[0]
print radar_text_train.shape[0]

radar_validation_idx = np.loadtxt(source_path + 'sources/sampling_triplets/radar/radar_validation_sample.txt')
radar_image_validation = np.load(data_path + 'radar_validation.npy')
radar_text_validation = np.loadtxt(source_path + 'labels/radar/radar_validation_full_labels.txt')
radar_val_num = radar_validation_idx.shape[0]
print radar_validation_idx.shape[0]
print radar_image_validation.shape[0]
print radar_text_validation.shape[0]


tradesy_train_idx = np.loadtxt(source_path + 'sources/sampling_triplets/tradesy/tradesy_train_sample.txt')
tradesy_image_train = np.load(data_path + 'tradesy_train.npy')
tradesy_text_train = np.loadtxt(source_path + 'labels/tradesy/tradesy_train_full_labels.txt')
tradesy_train_num = tradesy_train_idx.shape[0]
print tradesy_train_idx.shape[0]
print tradesy_image_train.shape[0]
print tradesy_text_train.shape[0]

tradesy_validation_idx = np.loadtxt(source_path + 'sources/sampling_triplets/tradesy/tradesy_validation_sample.txt')
tradesy_image_validation = np.load(data_path + 'tradesy_validation.npy')
tradesy_text_validation = np.loadtxt(source_path + 'labels/tradesy/tradesy_validation_full_labels.txt')
tradesy_val_num = tradesy_validation_idx.shape[0]
print tradesy_validation_idx.shape[0]
print tradesy_image_validation.shape[0]
print tradesy_text_validation.shape[0]

# train model
train_num = (chic_train_num + radar_train_num + tradesy_train_num) / batch_size
valid_num = (chic_val_num + radar_val_num + tradesy_val_num) / batch_size
print train_num
print valid_num

model.fit_generator(train_generator(),
                    steps_per_epoch = train_num,
                    epochs = 20,
                    verbose = 1,
                    validation_data = valid_generator(),
                    validation_steps = valid_num)

print('Writing model ...')
model.save('model.h5')
model_json = model.to_json()
file_name = 'model.json'
with open(file_name, "w") as json_file:
        json_file.write(model_json)

