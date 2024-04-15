import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

def avg_mut_info(matrix, label, N=10):
    """
    Mutual information score for each feature
    -------------------------
    Input: 
    - matrix (pd.DataFrame) : sample X feature matrix
    - label (dict) : sample label

    Output:
    - score (pd.DataFrame) : average mutual information score for each feature
    """
    # sampling N times
    score = pd.DataFrame(np.zeros(matrix.shape[1]),index=matrix.columns)[0]
    for i in range(1,N+1):
        obj = matrix_sampling(matrix,label)
        score += obj.mutual_info()
    return (score/N)

class matrix_sampling:
    def __init__(self, matrix, label):
        self.data = matrix
        self.label = label
        self.num_classes = len(set(label.values()))
        self.counter_classes = collections.Counter(label.values())
        self.sampling_and_rank()

    def sampling_and_rank(self):
        N_samples = min(self.counter_classes.values())

        # Eqaulty sampling for each class
        samples_idx = []
        for key in self.counter_classes.keys():
            ids_tmp = filter(lambda x:x is not None, map(lambda x:x if self.label[x]==key else None, self.data.index))
            samples_idx += np.random.choice(list(ids_tmp),N_samples,replace=False).tolist()
        self.data_sampled = self.data.loc[samples_idx,:]
        self.binned_data_sampled = self.data_sampled.apply(lambda x: pd.qcut(-x,self.num_classes,labels=False),axis=0)

    def mutual_info(self):
        labels = self.binned_data_sampled.index.map(lambda x:self.label[x])
        le = LabelEncoder()
        crit = le.fit_transform(labels)
        return self.binned_data_sampled.apply(lambda x: self.jointProb(x,crit),axis=0)

    def jointProb(self, vector, crit):
        if np.array(crit).shape != np.array(vector).shape:
            print('Dimension should be same')
        prob_mat = np.ones((self.num_classes, self.num_classes))
        locs = np.vstack((np.array(crit), np.array(vector))).T
        for idx in locs:
            prob_mat[tuple(idx)] += 1
        prob_mat/=prob_mat.sum()
        mut_info_mat = np.log(prob_mat * self.num_classes**2)*prob_mat
        return mut_info_mat.sum()