from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import edward as ed
import numpy as np
import six

import train_utils as util
from edward.models import Normal,Bernoulli,Categorical,Gamma,TransformedDistribution,RandomVariable


class MultiClass_BPM:
    def __init__(self,N,B,H,M,x_ph,y_ph):
        self.N = N  # number of training data
        self.B = B  # batch size
        self.H = H  # number of classes
        self.M = M  # number of features
        self.x_ph = x_ph
        self.y_ph = y_ph
    
    def define_model(self,weights_mean_prior=None,weights_precision_prior=None,noisePrecisionPrior=None):
        ds = tf.contrib.distributions 
        H = self.H
        M = self.M
        if noisePrecisionPrior==None:
            self.noise_p = Gamma(tf.ones([1]),tf.ones([1]))
        else:
            if isinstance(noisePrecisionPrior,RandomVariable):
                self.noise_p = noisePrecisionPrior
            else:
                raise TypeError('Not supported prios type!')
        self.q_noise_p = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([1])), \
                                scale=tf.nn.softplus(tf.Variable(tf.random_normal([1])))),bijector=ds.bijectors.Exp())
        
        if weights_mean_prior==None:
            self.w_mean = tf.zeros([H,M])  
            self.qw_mean = tf.Variable(tf.random_normal([H,M]))          
        else:               
            if isinstance(weights_mean_prior,(Normal,int,float,np.ndarray)):
                self.w_mean = weights_mean_prior 
                if isinstance(weights_mean_prior,Normal):
                    #print('mean prior normal')
                    self.qw_mean = Normal(tf.Variable(tf.random_normal([H,M])), tf.nn.softplus(tf.Variable(tf.random_normal([H,M]))))
                else:
                    #print('mean prior scalor')
                    self.qw_mean = tf.Variable(tf.random_normal([H,M]))
            else:
                raise TypeError('Not supported prios type!')

        if weights_precision_prior==None:
            self.w_precision = tf.ones([H,M])
            self.qw_precision = tf.nn.softplus(tf.Variable(tf.random_normal([H,M])))
        else:
            if isinstance(weights_precision_prior,(Gamma,int,float,np.ndarray)):
                self.w_precision = weights_precision_prior
                if isinstance(weights_precision_prior,Gamma):
                    self.qw_precision = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([H,M])), \
                                scale=tf.nn.softplus(tf.Variable(tf.random_normal([H,M])))),bijector=ds.bijectors.Exp())
                else:
                    self.qw_precision = tf.nn.softplus(tf.Variable(tf.random_normal([H,M])))
            else:
                raise TypeError('Not supported prios type!')

        self.w = Normal(self.w_mean,1./tf.sqrt(self.w_precision))
        #self.w = Normal(self.w_mean,(self.w_precision))
        #self.qw = Normal(tf.Variable(tf.random_normal([H,M])), tf.nn.softplus(tf.Variable(tf.random_normal([H,M]))))
        #self.qw = Normal(self.qw_mean, self.qw_precision)
        self.qw = Normal(self.qw_mean, 1./tf.sqrt(self.qw_precision))
        self.y = Categorical(tf.nn.softmax(Normal(tf.matmul(self.x_ph,tf.transpose(self.w)), 1./tf.sqrt(self.noise_p))))
        #self.y_test = tf.nn.softmax(tf.matmul(x_ph,tf.transpose(qw.loc)))
        
     
    def init_inference(self,infer_type='VI',niter=1000,nprint=100):
        
        if infer_type != 'VI':
            raise NotImplementedError('Not implemented this inference type yet!')
        self.infer_type = infer_type
        
        self.scaling = float(self.N) / self.B
        self.niter = niter
        self.nprint = nprint
        latent_vars = {self.w:self.qw,self.noise_p:self.q_noise_p}
        if infer_type == 'EP':
            self.inference = ed.KLpq({self.w:self.qw,self.noise_p:self.q_noise_p},data={self.y:self.y_ph})
        elif infer_type == 'VI':
            if isinstance(self.w_mean,Normal):
                latent_vars[self.w_mean] = self.qw_mean
            if isinstance(self.w_precision,Gamma):
                latent_vars[self.w_precision] = self.qw_precision
            self.inference = ed.KLqp(latent_vars,data={self.y:self.y_ph})
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
            '''
            if t % self.nprint == 0:
                print('\n w mean:')
                print(np.mean(self.sess.run(self.qw)))
            '''
                
    def predict(self, X_test, Y_test, proba=False):
        
        x_test_ph = tf.placeholder(tf.float32, [X_test.shape[0],X_test.shape[1]])
        y_test = tf.nn.softmax(tf.matmul(x_test_ph,tf.transpose(self.qw.loc)))
        ii = 0
        acu = 0
        N_test = X_test.shape[0]
        y_predict = np.argmax(self.sess.run(y_test,feed_dict={x_test_ph:X_test}),axis=1)
        acu = sum(y_predict==Y_test)
        
        print('Test accuracy: ', acu*1./N_test)
        return y_predict
    