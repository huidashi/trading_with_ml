import numpy as np


def test_run():
    a = np.random.randint(0,10,size=(5,4))  # 5x4 array of random numbers
    # print(a)
    # print(a.shape)
    #
    # print(a.shape[0])  # number of rows
    # print(a.shape[1])  # number of columns

    print(len(a.shape))
    print(a.size)
    print(a.dtype)


if __name__ == "__main__":
    test_run()