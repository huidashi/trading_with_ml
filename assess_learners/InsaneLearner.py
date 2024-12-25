import BagLearner as bg, LinRegLearner as lrl, numpy as np
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.learners = []
        self.verbose = verbose
        for i in range(20):
            self.learners.append(bg.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, verbose = False, boost = False))
    def author(self):
        return "hshi320"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        pred = []
        for learner in self.learners:
            pred.append(learner.query(points))
        return np.mean(pred, axis = 0)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")