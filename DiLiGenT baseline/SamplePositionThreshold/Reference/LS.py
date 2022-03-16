import numpy as np
import scipy.io
import cv2
import re

def L2_PMS(data, m,lowPerc,highPerc) :

    m = np.reshape(m, -1).astype(np.int32)
    light_dir = np.transpose(data['s']) # 매트랩과 같다.
    f = light_dir.shape[1]

    height, width, color = data['mask'].shape
    p = len(m)

    I = np.zeros((p, f)) # (15791, 96)

    for i in range(f) : # frame
        img = data['imgs'][i]
        # 0.299 * R + 0.587 * G + 0.114
        img = np.add(np.add(0.299*img[:,:,0], 0.587*img[:,:,1]), 0.114*img[:,:,2]).astype(np.float64) # 여기서 512*612*3 -> 512*612
        # gray scale을 구하는 과정에서 차이가 발생 -> 오차들이 하나둘씩 모여 I에서 큰 차이 발생
        I[:, i] = (np.reshape(np.transpose(img), -1)[m]).copy()
    
    L = light_dir.copy()

#######################################################################
############################# MODIFIED FROM HERE#######################
    S_hat = np.dot(I, np.linalg.pinv(L))

    S = np.zeros(S_hat.shape)

    signX = 1
    signY = 1
    signZ = 1

    for i in range(p) :
        length = np.round(np.sqrt( S_hat[i, 0]*S_hat[i, 0] + S_hat[i, 1]*S_hat[i, 1] + S_hat[i, 2]*S_hat[i, 2]), 4)
        S[i, 0] = np.multiply(np.divide(S_hat[i, 0] ,length), signX)
        S[i, 1] = np.multiply(np.divide(S_hat[i, 1] ,length), signY)
        S[i, 2] = np.multiply(np.divide(S_hat[i, 2] ,length), signZ)

######################################################################
    n_x = np.resize(np.zeros(1), (height*width, 1)) # (313344,1)
    n_y = np.resize(np.zeros(1), (height*width, 1))
    n_z = np.resize(np.zeros(1), (height*width, 1))

    # 법선 벡터. 여기 통과
    for i in range(p):
        n_x[m[i]] = S[i, 0].copy()
        n_y[m[i]] = S[i, 1].copy()
        n_z[m[i]] = S[i, 2].copy()

    n_x = np.reshape(n_x, -1)
    n_y = np.reshape(n_y, -1)
    n_z = np.reshape(n_z, -1)

    _N = np.zeros((height*width, 3))

    _N[: ,0] = np.transpose(n_x.copy())
    _N[:, 1] = np.transpose(n_y.copy())
    _N[:, 2] = np.transpose(n_z.copy())

    # Transpose를 하니까 (B, G, R)순으로, 다시 말해 법선 벡터의 (Z, Y, X)순으로 넣어준다.
    N = np.transpose(np.array([np.reshape(_N[: ,2], (width, height)), np.reshape(_N[: ,1], (width, height)), np.reshape(_N[: ,0], (width, height))]))  

    N[np.isnan(N)] = 0

    return N,scount
