import numpy as np
import random
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner, kwargs={}, bags=20, verbose=False, boost = False):
        """
        Constructor method
        """
        self.boost = boost
        self.learners = []
        self.bags = bags
        self.verbose = verbose

        for i in range(self.bags):
            self.learners.append(learner(**kwargs))


    def author(self):

        return "hshi320"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):

        data= np.hstack((data_x,data_y.reshape(-1,1)))
        self.bagging(data)


    def bagging(self, data):
        n_samples = data.shape[0]
        data_y = data[:, -1]
        data_x = data[:, 0:-1]

        for learner in self.learners:
            bootstrap_index = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_x = data_x[bootstrap_index]
            bootstrap_y = data_y[bootstrap_index]

            learner.add_evidence(bootstrap_x, bootstrap_y)

    def query(self, points):
        pred = []
        for learner in self.learners:
            pred.append(learner.query(points))
        return np.mean(pred, axis = 0)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

