""""""
from seaborn.conftest import flat_series
import matplotlib.pyplot as plt
"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "hshi320"  # replace tb34 with your Georgia Tech username.
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return 904069365  # replace with your GT ID number
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    result = False  		  	   		 	   		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	   		  		  		    	 		 		   		 		  
        result = True  		  	   		 	   		  		  		    	 		 		   		 		  
    return result  		  	   		 	   		  		  		    	 		 		   		 		  

def balch_episode(run=10, win_prob=(18/38), bankroll_start=0, win_limit = 80):
    if bankroll_start == 0:
        episode_array = np.full((run,1000), 0)
        for x in range(run):
            episode_winnings = 0
            i=0
            while i < 1000 and episode_winnings < win_limit:
                won = False
                bet_amount = 1
                while not won:
                    won = get_spin_result(win_prob)
                    if won == True:
                        episode_winnings = episode_winnings + bet_amount
                        if episode_winnings >= win_limit:
                            episode_array[x, i:] = win_limit
                            break
                        episode_array[x,i] = episode_winnings
                    else:
                        episode_winnings = episode_winnings - bet_amount
                        bet_amount = bet_amount *2
                        episode_array[x, i] = episode_winnings

                    i = i + 1
            x = x + 1
        return episode_array
    else:
        episode_array = np.full((run,1000), win_limit)
        for x in range(run):
            bet_amount = 1
            episode_winnings = 0
            i = 0
            bankroll = bankroll_start
            while i < 1000 and episode_winnings < win_limit:
                won = get_spin_result(win_prob)
                if won == True:
                    episode_winnings = episode_winnings + bet_amount
                    bankroll = bankroll + bet_amount
                    episode_array[x,i] = episode_winnings
                    bet_amount = 1
                else:
                    episode_winnings = episode_winnings - bet_amount
                    bankroll = bankroll - bet_amount
                    bet_amount = bet_amount *2
                    if bankroll == 0 or bankroll < 0:
                        episode_array[x,i:] = -1.0 * bankroll_start
                        break
                    elif bet_amount >= bankroll:
                        bet_amount = bankroll
                    episode_array[x, i] = episode_winnings
                i = i + 1
                #print (bankroll)
        return episode_array






def plot_fig_1():
    array = balch_episode(run=10)

    plt.title('fg.1 10 episode infinite bankroll')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    for i in range(10):
        plt.plot(array[i], label='episode' + str(i+1))
    plt.legend()
    plt.savefig('figure_1.png')
    plt.show()
    plt.clf()

def plot_fig_2():
    array = balch_episode(run=1000)

    plt.title('fg.2 1000 episode mean infinite bankroll')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    mean = np.mean(array, axis=0)
    std = array.std(axis=0)

    plt.plot(mean, label='mean')
    plt.plot(mean+std, label='mean+std')
    plt.plot(mean-std, label='mean-std')
    plt.legend()
    plt.savefig('figure_2.png')
    plt.show()
    plt.clf()

def plot_fig_3():
    array = balch_episode(run=1000, win_limit = 80)

    plt.title('fg.3 1000 episode median infinite bankroll')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    median = np.median(array, axis=0)
    std = array.std(axis=0)

    plt.plot(median, label='median')
    plt.plot(median+std, label='median+std')
    plt.plot(median-std, label='median-std')
    plt.legend()
    plt.savefig('figure_3.png')
    plt.show()
    plt.clf()

def plot_fig_4():
    array = balch_episode(run = 1000,bankroll_start = 256)

    plt.title('fg.4 1000 episode mean 256 bankroll')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    mean = np.mean(array, axis=0)
    std = array.std(axis=0)

    plt.plot(mean, label='mean')
    plt.plot(mean+std, label='mean+std')
    plt.plot(mean-std, label='mean-std')
    plt.legend()
    plt.savefig('figure_4.png')
    plt.show()
    plt.clf()

def plot_fig_5():
    array = balch_episode(run = 1000,bankroll_start = 256)

    plt.title('fg.5 1000 episode median 256 bankroll')
    plt.axis([0, 300, -256, 100])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    median = np.median(array, axis=0)
    std = array.std(axis=0)

    plt.plot(median, label='median')
    plt.plot(median+std, label='median+std')
    plt.plot(median-std, label='median-std')
    plt.legend()
    plt.savefig('figure_5.png')
    plt.show()
    plt.clf()

def plot_fig_6():
    array = balch_episode(run=1000, win_limit = 200)

    plt.title('fg.6 1000 episode infinite bankroll 200 win limit')
    plt.axis([0, 800, -500, 500])
    plt.xlabel('Bet #')
    plt.ylabel('Winnings')

    mean = np.mean(array, axis=0)
    std = array.std(axis=0)

    plt.plot(mean , label='mean')
    plt.plot(mean+std, label='mean+std')
    plt.plot(mean-std, label='mean-std')
    plt.legend()
    plt.savefig('figure_6.png')
    plt.show()
    plt.clf()


def happy_ratio(array):
    happy = 0
    sad = 0
    for i in range(len(array)):
        if array[i,-1] == 80:
            happy = happy + 1
        else:
            sad = sad + 1
    return ('probability of winning: ' + str(happy/(len(array[0])+1)*100) +
            '% | probability of losing: ' + str(sad/(len(array[0])+1)*100) +'%')
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    win_prob = (18/38)  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		 	   		  		  		    	 		 		   		 		  
    print(balch_episode(1000,win_prob,256)[0])  # test the roulette spin
    # add your code here to implement the experiments
    plot_fig_1()
    plot_fig_2()
    plot_fig_3()
    plot_fig_4()
    plot_fig_5()
    plot_fig_6()

    # print (happy_ratio(balch_episode(run=1000, win_prob=(18/38), bankroll_start=256, win_limit = 80)))

if __name__ == "__main__":
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
