from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import six
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_next_batch(data, B, ii,labels=None):
    if ii+B < data.shape[0]:
        if labels!=None:
            return data[ii:ii+B],labels[ii:ii+B],ii+B
            
        else:
            return data[ii:ii+B],labels[ii:ii+B],ii+B
    else:
        r = ii+B-data.shape[0]
        ids = np.array(range(data.shape[0]))
        batch = data[(ids>=ii)|(ids<r)]
        if labels == None:
            return batch,labels,r
        else:
            return batch,labels[(ids>=ii)|(ids<r)],r

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def standardize(x,each_feature=False):
    if isinstance(x,tf.Tensor):
        if each_feature:
            d = []
            for i in range(x.get_shape()[1].value):
                m = tf.reduce_mean(x[:,i])
                v = tf.reduce_mean(tf.square(x[:,i]-m))
                if v==0:
                    v=1.
                d.append(tf.reshape((x[:,i]-m)/tf.sqrt(v), [-1,1]))
            return tf.concat(d,1)
                
        m = tf.reduce_mean(x)
        v = tf.reduce_mean(tf.square(x-m))
        return (x-m)/tf.sqrt(v)
    else:
        if each_feature:
            for i in range(x.shape[1]):
                v = np.std(x[:,i])
                if v==0:
                    v=1.
                x[:,i] = x[:,i] - np.mean(x[:,i])/v
            return x
        return (x-np.mean(x))/np.std(x)


def build_toy_dataset(mtype,N,M,K,s_std=1,s_mean=0,d_std=1,d_mean=0,noise_std=0.1):
    if mtype == 'dictionary':
        D = []
        #print(s_mean,s_std)
        s = np.random.normal(s_mean,s_std,(N,K))               
        for k in range(K):
            D.append(np.sin(((k+1)*2*(np.arange(0,2*np.pi,2*np.pi/M)))))
        D = np.vstack(D)
        x = np.matmul(s,D) + np.random.normal(0, noise_std, size=(N,M))
        return (D,x)
    elif mtype == 'gaussian':
        s = np.random.normal(s_mean,s_std,(N,M))
        x = s + np.random.normal(0,noise_std,size=(N,M))
        return (None,x)
    elif mtype == 'bpm':
        y = np.random.choice(K,N)
        x = np.zeros((N,M))
        s_mean = np.zeros(M) + s_mean
        s_std = np.ones(M) * s_std
        for k in range(K):
            mean = s_mean+k*d_std
            ids = (y==k)
            x[ids] = np.random.normal(mean,s_std,(sum(ids),M))       
        return (y,x)


def one_hot_encoder(label, N, H):
    Y = np.zeros((N,H))
    Y[range(N),label] = 1
    return Y
