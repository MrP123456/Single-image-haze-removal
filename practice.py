import numpy as np

if __name__ == '__main__':
    a = np.array([1,2,3])
    b=a.reshape([1,1,-1])
    print(b.shape)
