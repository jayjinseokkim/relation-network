import json
import numpy as np
import os
from PIL import Image
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 50
tokenizer = Tokenizer()


def load_data(path):
    f = open(path, 'r')
    data = []
    for l in f:
        jn = json.loads(l)
        s = jn['sentence']
        idn = jn['identifier']
        la = int(jn['label'] == 'true')
        data.append([idn, s, la])
    return data


def tokenize_data(sdata, mxlen):
    texts = [t[1] for t in sdata]
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = pad_sequences(seqs, mxlen)
    data = {}
    for k in range(len(sdata)):
        data[sdata[k][0]] = [seqs[k], sdata[k][2]]
    return data


def load_images(path, sdata, debug=False):
    data = {}
    cnt = 0
    N = 1000
    for lists in os.listdir(path):
        p = os.path.join(path, lists)
        for f in os.listdir(p):
            cnt += 1
            if debug and cnt > N:
                break
            im_path = os.path.join(p, f)
            im = Image.open(im_path)
            im = im.convert('RGB')
            im = im.resize((200, 50))
            im = np.array(im)
            idf = f[f.find('-') + 1:f.rfind('-')]
            data[f] = [im] + sdata[idf]
    ims, ws, labels = [], [], []
    for key in data:
        ims.append(data[key][0])
        ws.append(data[key][1])
        labels.append(data[key][2])
    data.clear()
    idx = np.arange(0, len(ims), 1)
    np.random.shuffle(idx)
    ims = [ims[t] for t in idx]
    ws = [ws[t] for t in idx]
    labels = [labels[t] for t in idx]
    ims = np.array(ims, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return ims, ws, labels


def get_embeddings_index():
    embeddings_index = {}
    path = r'C:\local\word2vec\glove.6B.50d.txt'
    f = open(path, 'r', errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def embedding_layer(word_index, embedding_index, sequence_len):
    embedding_matrix = get_embedding_matrix(word_index, embedding_index)
    return Embedding(len(word_index) + 1,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=sequence_len,
                     trainable=False)