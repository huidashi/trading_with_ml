import numpy as np


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):

        return "hshi320"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):

        data= np.hstack((data_x,data_y.reshape(-1,1)))
        self.tree = self.build_tree(data)

    def cal_corr(self, feature, target):
        corr_matrix = np.corrcoef(feature, target)
        return abs(corr_matrix[0,1])

    def best_split(self, data):
        data_y = data[:,-1]
        best_feature_index = None
        best_corr = 0

        for i in range(data.shape[1]-1):
            feature = data[:,i]
            corr = self.cal_corr(feature, data_y)
            if corr > best_corr:
                best_feature_index = i
                best_corr = corr
        split_val = np.median(data[:, best_feature_index])
        return split_val, best_feature_index

    def build_tree(self,data):
        data_y = data[:, -1]
        data_x = data[:, 0:-1]

        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y),-1, -1]], dtype=float)
        elif np.all(data_y == data_y[0]):
            return np.array([[-1, np.mean(data_y), -1, -1]], dtype=float)

        split_val, best_feature_index = self.best_split(data)
        if split_val == np.max(data[:,best_feature_index]) or split_val == np.min(data[:,best_feature_index]):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        left_data = data[data[:, best_feature_index] <= split_val]
        right_data = data[data[:, best_feature_index] > split_val]
        left_tree = self.build_tree(left_data)
        right_tree = self.build_tree(right_data)

        root = np.array([[best_feature_index, split_val, 1, left_tree.shape[0]+1]], dtype=float)
        decision_tree = np.vstack((root, left_tree, right_tree))
        return decision_tree

    def query(self, points):
        pred = []
        for x in points:
            node = 0
            while self.tree[node,0] !=-1:
                feature_index = int(self.tree[node,0])
                split_val = self.tree[node,1]

                #left
                if x[feature_index] <= split_val:
                    node = node +1
                #right
                else:
                    node = node + int(self.tree[node,-1])
            decision = self.tree[node, 1]
            factor = self.tree[node, 0]
            pred.append(decision)
        return pred
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")

