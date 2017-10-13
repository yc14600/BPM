from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import edward as ed
import numpy as np
import six

import train_utils as util
from edward.models import Normal,Bernoulli,Categorical,Gamma,TransformedDistribution


class MultiClass_BPM:
    def __init__(self,N,B,H,M,x_ph,y_ph,noisePrecisionPriorParams=(1,1)):
        self.N = N  # number of training data
        self.B = B  # batch size
        self.H = H  # number of classes
        self.M = M  # number of features
        self.x_ph = x_ph
        self.y_ph = y_ph
        
        ds = tf.contrib.distributions 
        self.noise_p = Gamma(tf.ones(1)*noisePrecisionPriorParams[0],tf.ones(1)*noisePrecisionPriorParams[1])
        self.q_noise_p = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([1])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([1])))),bijector=ds.bijectors.Exp())
        self.w = Normal(tf.zeros([H,M]),tf.ones([H,M]))
        self.qw = Normal(tf.Variable(tf.random_normal([H,M])), tf.nn.softplus(tf.Variable(tf.random_normal([H,M]))))
        self.y = Categorical(tf.nn.softmax(Normal(tf.matmul(x_ph,tf.transpose(self.w)), 1./tf.sqrt(self.noise_p))))
        #self.y_test = tf.nn.softmax(tf.matmul(x_ph,tf.transpose(qw.loc)))
        
     
    def init_inference(self,infer_type='VI',niter=1000,nprint=100):
        
        if infer_type != 'VI':
            raise NotImplementedError('Not implemented this inference type yet!')
        self.infer_type = infer_type
        
        self.scaling = float(self.N) / self.B
        self.niter = niter
        self.nprint = nprint

        if infer_type == 'EP':
            self.inference = ed.KLpq({self.w:self.qw,self.noise_p:self.q_noise_p},data={self.y:self.y_ph})
        elif infer_type == 'VI':
            self.inference = ed.KLqp({self.w:self.qw,self.noise_p:self.q_noise_p},data={self.y:self.y_ph})
        elif infer_type == 'HMC':
            self.inference = ed.HMC({self.w:self.qw},data={self.y:self.y_ph})
        else:
            raise NotImplementedError('Not implemented this inference type yet!')

        self.inference.initialize(n_iter=niter,n_print=nprint,scale={self.y:self.scaling})
    
    def fit(self,X,Y):
        
        self.sess = ed.get_session()
        tf.global_variables_initializer().run()
        ii = 0
        for t in range(self.niter):
            x_batch,y_batch,ii = util.get_next_batch(X,self.B,ii,Y)

            info_dict = self.inference.update(feed_dict={self.x_ph:x_batch,self.y_ph:y_batch})
            self.inference.print_progress(info_dict)

            if t % self.nprint == 0:
                print('\n w mean:')
                print(np.mean(self.sess.run(self.qw)))
                
    def predict(self, X_test, Y_test, proba=False):
        y_test = tf.nn.softmax(tf.matmul(self.x_ph,tf.transpose(self.qw.loc)))
        ii = 0
        acu = 0
        N_test = X_test.shape[0]
        y_predict = np.zeros(N_test)
        for i in range(int(np.floor(N_test/self.B))):
            x_batch,y_batch,ii = util.get_next_batch(X_test,self.B,ii,Y_test)
            y_test_batch = self.sess.run(y_test,feed_dict={self.x_ph:x_batch,self.y_ph:y_batch})
            if proba:
                y_predict[ii:ii+self.B] = y_test_batch
            else:
                y_predict[ii:ii+self.B] = np.argmax(y_test_batch,axis=1)
                
            acu += sum(np.argmax(y_test_batch,axis=1)==y_batch)
        print('Test accuracy: ', acu*1./N_test)
        return y_predict
    