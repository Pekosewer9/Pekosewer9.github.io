import numpy as np
import os
class NearestNeighbor(object):
  def __init__(self):
    pass

    def train(self, X, y):
        self.Xtr = X  # N * D
        self.Ytr = y # labels
    
    def predict(self, X):
        # X [N * D]
        # using L1
        N = X.shape[0] 
        y_predict = np.zeros(N,dtype = self.Ytr.dtype)
        # para1 = N , dtype 指定数据类型
        for i in range(N):
            distances = np.sum(np.abs(X[i:] - self.Xtr), axis = 1) # index = [N,]
            index =  np.argmin(distances)
            y_predict[i] = self.Ytr[index]

        return y_predict
    
    '''
    np.zeros()
    np.sum(para,axis = x)
    np.argmin() 返回index
    
    '''