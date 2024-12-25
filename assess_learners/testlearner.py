""""""
from matplotlib import pyplot as plt

"""  		  	   		 	   		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bg
import InsaneLearner as ins
import sys
import math
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		 	   		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		  	   		 	   		  		  		    	 		 		   		 		  
    data = np.array(  		  	   		 	   		  		  		    	 		 		   		 		  
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )
    np.random.shuffle(data)
    # compute how much of the data is training and testing  		  	   		 	   		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	   		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		 	   		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		 	   		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		 	   		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    #experiment 1
    in_sample_rmse_dt = []
    out_sample_rmse_dt = []
    in_sample_rmse_rt = []
    out_sample_rmse_rt = []
    for i in range (1,41):
        learner =dt.DTLearner(leaf_size = i, verbose=True)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)  # get the predictions
        rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample_rmse_dt.append(rmse_train)

        pred_y = learner.query(test_x)  # get the predictions
        rmse_pred = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample_rmse_dt.append(rmse_pred)

        learner2 = rt.RTLearner(leaf_size = i, verbose=True)
        learner2.add_evidence(train_x, train_y)

        pred_y = learner2.query(train_x)  # get the predictions
        rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample_rmse_rt.append(rmse_train)

        pred_y = learner2.query(test_x)  # get the predictions
        rmse_pred = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample_rmse_rt.append(rmse_pred)

    plt.plot(in_sample_rmse_dt)
    plt.plot(out_sample_rmse_dt)

    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.xlim(0,40)
    plt.ylim(0,0.01)
    plt.title('DTLearner RMSE per Leaf Size')
    plt.legend(['In Sample DT', 'Out Sample DT'])
    plt.savefig('Fig_1_DTLearner_RMSE.png')
    plt.clf()
    plt.title('RTLearner RMSE per Leaf Size')
    plt.legend(['In Sample RT', 'Out Sample RT'])
    plt.plot(in_sample_rmse_rt)
    plt.plot(out_sample_rmse_rt)
    plt.savefig('Fig_2_RTLearner_RMSE.png')
    plt.clf()

    # experiment 2
    in_sample_rmse_bg = []
    out_sample_rmse_bg = []
    for i in range(1, 41):
        learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=True)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)  # get the predictions
        rmse_train = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample_rmse_bg.append(rmse_train)

        pred_y = learner.query(test_x)  # get the predictions
        rmse_pred = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample_rmse_bg.append(rmse_pred)

    plt.plot(in_sample_rmse_bg)
    plt.plot(out_sample_rmse_bg)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.xlim(0, 40)
    plt.ylim(0,0.01)
    plt.title('BagLearner(using DTLearner) RMSE per Leaf Size')
    plt.legend(['In Sample BagLearner(DT)', 'Out Sample BagLearner(DT)'])
    plt.savefig('Fig_3_BagLearner_RMSE.png')
    plt.clf()

    #experiment 3
    in_sample_mae_dt = []
    out_sample_mae_dt = []
    in_sample_mae_rt = []
    out_sample_mae_rt = []

    in_sample_r2_dt = []
    out_sample_r2_dt = []
    in_sample_r2_rt = []
    out_sample_r2_rt = []
    for i in range(1, 41):
        #DTLearner
        learner = dt.DTLearner(leaf_size=i, verbose=True)
        learner.add_evidence(train_x, train_y)

        #mae
        pred_y = learner.query(train_x)
        mae_train = np.mean(np.abs(train_y - pred_y))  # calculate MAE
        in_sample_mae_dt.append(mae_train)

        #r2
        sum_square_residuals = ((train_y - pred_y)**2).sum()
        sum_square_total = (train_y - np.mean(train_y)**2).sum()
        r2_train = 1- (sum_square_residuals / sum_square_total)
        in_sample_r2_dt.append(r2_train)

        #mae
        pred_y = learner.query(test_x)
        mae_pred = np.mean(np.abs(test_y - pred_y))
        out_sample_mae_dt.append(mae_pred)

        #r2
        sum_square_residuals = ((test_y - pred_y)**2).sum()
        sum_square_total = (test_y - np.mean(test_y)**2).sum()
        r2_pred = 1- (sum_square_residuals / sum_square_total)
        out_sample_r2_dt.append(r2_pred)

        # RTLearner

        learner2 = rt.RTLearner(leaf_size=i, verbose=True)
        learner2.add_evidence(train_x, train_y)

        pred_y = learner2.query(train_x)
        mae_train = np.mean(np.abs(train_y - pred_y))  # calculate MAE
        in_sample_mae_rt.append(mae_train)

        #r2
        sum_square_residuals = ((train_y - pred_y)**2).sum()
        sum_square_total = (train_y - np.mean(train_y)**2).sum()
        r2_train = 1- (sum_square_residuals / sum_square_total)
        in_sample_r2_rt.append(r2_train)

        #mae
        pred_y = learner2.query(test_x)
        mae_pred = np.mean(np.abs(test_y - pred_y))
        out_sample_mae_rt.append(mae_pred)

        #r2
        sum_square_residuals = ((test_y - pred_y)**2).sum()
        sum_square_total = (test_y - np.mean(test_y)**2).sum()
        r2_pred = 1- (sum_square_residuals / sum_square_total)
        out_sample_r2_rt.append(r2_pred)

    plt.plot(in_sample_mae_dt)
    plt.plot(out_sample_mae_dt)
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE')
    plt.xlim(0, 40)
    plt.ylim(0, 0.01)
    plt.title('DTLearner MAE per Leaf Size')
    plt.legend(['In Sample DT MAE', 'Out Sample DT MAE'])
    plt.savefig('Fig_4_DTLearner_MAE.png')
    plt.clf()
    plt.plot(in_sample_mae_rt)
    plt.plot(out_sample_mae_rt)
    plt.xlabel('Leaf Size')
    plt.ylabel('MAE')
    plt.xlim(0, 40)
    plt.ylim(0, 0.01)
    plt.title('RTLearner MAE per Leaf Size')
    plt.legend(['In Sample RT MAE', 'Out Sample RT MAE'])
    plt.savefig('Fig_5_RTLearner_MAE.png')
    plt.clf()

    plt.plot(in_sample_r2_dt)
    plt.plot(out_sample_r2_dt)
    plt.xlabel('Leaf Size')
    plt.ylabel('R2')
    plt.xlim(0, 40)
    plt.ylim(0, 1)
    plt.title('DTLearner R-Squared per Leaf Size')
    plt.legend(['In Sample DT R-Squared', 'Out Sample DT R-Squared'])
    plt.savefig('Fig_6_DTLearner_R2.png')
    plt.clf()
    plt.plot(in_sample_r2_rt)
    plt.plot(out_sample_r2_rt)
    plt.xlabel('Leaf Size')
    plt.ylabel('R2')
    plt.xlim(0, 40)
    plt.ylim(0, 1)
    plt.title('RTLearner R-Squared per Leaf Size')
    plt.legend(['In Sample RT R-Squared', 'Out Sample RT R-Squared'])
    plt.savefig('Fig_7_RTLearner_R2.png')
    plt.clf()

    # learner = bg.BagLearner(learners=dt.DTLearner, kwargs={"leaf_size": 5}, bags=10, boost=False, verbose=False)# create a LinRegLearner
    # learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())
    # print(learner.bagging(data))


    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
    #
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
