import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Embedding,\
    LSTM, Bidirectional, Lambda, Concatenate, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import gc
import prepare
import subprocess

mxlen = 32
embedding_dim = 50
lstm_unit = 128
MLP_unit = 128
epochs = 100

train_json = 'nlvr\\train\\train.json'
train_img_folder = 'nlvr\\train\\images'
data = prepare.load_data(train_json)
data = prepare.tokenize_data(data, mxlen)
imgs, ws, labels = prepare.load_images(train_img_folder, data, debug=True)
data.clear()
imgs_mean = np.mean(imgs)
imgs_std = np.std(imgs - imgs_mean)
imgs = (imgs - imgs_mean) / imgs_std

epochs = 100
batch_size = 64


def bn_layer(x, conv_unit):
    def f(inputs):
        md = Conv2D(x, (conv_unit, conv_unit), padding='same')(inputs)
        md = BatchNormalization()(md)
        return Activation('relu')(md)
    return f


def conv_net(inputs):
    model = bn_layer(32, 3)(inputs)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(32, 3)(model)
    model = MaxPooling2D((2, 2), 2)(model)
    model = bn_layer(64, 3)(model)
    return model


input1 = Input((50, 200, 3))
input2 = Input((mxlen,))
cnn_features = conv_net(input1)
embedding_layer = prepare.embedding_layer(prepare.tokenizer.word_index, prepare.get_embeddings_index(), mxlen)
embedding = embedding_layer(input2)
bi_lstm = Bidirectional(LSTM(lstm_unit, implementation=2, return_sequences=False))
lstm_encode = bi_lstm(embedding)
shapes = cnn_features.shape
w, h = shapes[1], shapes[2]
features = []
for k1 in range(w):
    for k2 in range(h):
        def get_feature(t):
            return t[:, k1, k2, :]
        get_feature_layer = Lambda(get_feature)
        features.append(get_feature_layer(cnn_features))

relations = []
concat = Concatenate()
for feature1 in features:
    for feature2 in features:
        relations.append(concat([feature1, feature2, lstm_encode]))


def get_dense(n):
    r = []
    for k in range(n):
        r.append(Dense(MLP_unit, activation='relu'))
    return r


def get_MLP(n, denses):
    def g(x):
        d = x
        for k in range(n):
            d = denses[k](d)
        return d
    return g


def dropout_dense(x):
    y = Dense(MLP_unit)(x)
    y = Dropout(0.5)(y)
    y = Activation('relu')(y)
    return y

g_MLP = get_MLP(4, get_dense(4))
f_MLP = get_MLP(2, get_dense(2))

mid_relations = []
for r in relations:
    mid_relations.append(g_MLP(r))
combined_relation = Add()(mid_relations)

rn = dropout_dense(combined_relation)
rn = dropout_dense(rn)
pred = Dense(1, activation='sigmoid')(rn)

model = Model(inputs=[input1, input2], outputs=pred)
optimizer = Adam(lr=3e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.fit([imgs, ws], labels, validation_split=0.1, epochs=epochs)
model.save('model')
gc.collect()
subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
