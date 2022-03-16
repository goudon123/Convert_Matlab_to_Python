import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt

from normal_img2vec import *
import util as ut



def main_Baseline(dataset_path) : 
    dataFormat = 'PNG'
    dataNameStack = np.array(['ball', 'cat', 'pot1', 'bear', 'pot2', 'buddha', 'goblet', 'reading', 'cow', 'harvest'])

#################################################################################################
###############################################################################################

    percLow = np.arange(0,0.50,0.05)
    percHigh = np.arange(1,0.50,-0.05)
    angErrMat = np.zeros((len(percLow),len(percHigh)))

###############################################################################################
###############################################################################################

    for testId in range(10) :
        dataName = dataNameStack[testId] +  dataFormat
        datadir = dataset_path + '/pmsData/' + dataName
        bitdepth = 16
        gamma = 1
        resize = 1
        data = ut.load_datadir_re(datadir, bitdepth, resize, gamma) 

        mask1 = 0
        if data['mask'].shape[2] == 1 :
            mask1 = data['mask']/255.0
        else : 
            mask1 = np.add(np.add(0.299*data['mask'][:,:,0], 0.587*data['mask'][:,:,1]), 0.114*data['mask'][:,:,2])/255.0 # (512, 612).
        
        mask3 = np.transpose(np.array([np.transpose(mask1), np.transpose(mask1), np.transpose(mask1)])) 

        m = np.reshape(np.array(np.argwhere(np.reshape(np.transpose(mask1), -1) == 1), dtype = object), -1) 


###############################################################################################
#######################    Load Ground Truth Normal   #########################################
#Load Ground Truth Normal

        mat_file_path =  resultDir + '/' + dataNameStack[iD] + dataFormat + '_Normal_' + methodStack[iM] + '.mat'
        Normal_est_mat_file = io.loadmat(mat_file_path)
        Normal_est = 0
        if iM == 0:
            Normal_est = Normal_est_mat_file['Normal_L2']
        else:
            Normal_est = Normal_est_mat_file['Normal_est']
            
        mat_file_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'Normal_gt.mat'
        Normal_gt_mat_file = io.loadmat(mat_file_path)
        Normal_gt = Normal_gt_mat_file['Normal_gt']

        mask_path = dataDir + '/' + dataNameStack[iD] + dataFormat + '/' + 'mask.png'
        img = cv2.imread(mask_path).astype(np.float64)
        mask = np.array(np.transpose( [np.transpose(img[:, :, 2]), np.transpose(img[:, :, 1]), np.transpose(img[:, :, 0]) ]))
        mask = np.add(np.add(0.299*mask[:,:,0], 0.587*mask[:,:,1]), 0.114*mask[:,:,2])
        th, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = np.array(mask)/255.0

        m = 0
        N_gt = 0
        m = np.reshape(np.array(np.argwhere(np.reshape(np.transpose(mask), -1) == 1), dtype = object), -1) 
        N_gt = normal_img2vec(Normal_gt, m)


###############################################################################################
#####################           MODIFIED FROM HERE(DONE)        ###############################

   # percLow 
   # percHigh
    for i in range(percLow) :
        for j in range(percHigh) :
            [N, scount] = L2_PMS_lowhigh_Allpixel(data, m, percLow(i), percHigh(j));


###############################################################################################
###############################################################################################

            N_est = normal_img2vec(N, m).astype(np.float64)

            dot_value = np.multiply(N_gt[:, 0], N_est[:, 0]) + np.multiply(N_gt[:, 1], N_est[:, 1]) + np.multiply(N_gt[:, 2], N_est[:, 2]) 

            angErr = []
            for i in range(len(dot_value)) :

                if abs(dot_value[i]) > 1.0 :
                    temp = 0
                    angErr.append(temp)
                else :
                    temp = np.real((np.arccos(dot_value[i]).astype(np.float64))) * (180.0 / np.pi)
                    angErr.append(temp)

            angErr = np.array(angErr)
            print('MeanErr-' + methodStack[iM] + '/' + dataNameStack[iD] + ":" + str(np.mean(angErr)))

            angErrMat[0:len(m), iD] = angErr;
        angErrStack.append(angErrMat)


###############################################################################################
###############################################################################################



def L2_PMS_lowhigh_Allpixel(dataset_path) : 

    m = np.reshape(m, -1).astype(np.int32)
    light_dir = np.transpose(data['s']) 
    f = light_dir.shape[1]

    height, width, color = data['mask'].shape
    p = len(m)

    I = np.zeros((p, f)) # (15791, 96)

    for i in range(f) : # frame
        img = data['imgs'][i]
        # 0.299 * R + 0.587 * G + 0.114
        img = np.add(np.add(0.299*img[:,:,0], 0.587*img[:,:,1]), 0.114*img[:,:,2]).astype(np.float64) #  512*612*3 -> 512*612

        I[:, i] = (np.reshape(np.transpose(img), -1)[m]).copy()
    
    L = light_dir.copy()

###############################################################################################
#####################             MODIFIED FROM HERE		  #############################

    S = np,zeros((p, 3))
    scount = np,zeros((p, 1))
    Ivec = I[:]
    Ival = np.sort(Ivec)
    
    lowTHval = Ival(np.floor(lowPerc * p* f)+1 )
    highTHval = Ival(np.floor(highPerc * p* f) )

    for i in range(p):
        I_p1 = I(i_row, :)
 


















###############################################################################################
###############################################################################################


    n_x = np.resize(np.zeros(1), (height*width, 1)) # (313344,1)
    n_y = np.resize(np.zeros(1), (height*width, 1))
    n_z = np.resize(np.zeros(1), (height*width, 1))


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

    
    N = np.transpose(np.array([np.reshape(_N[: ,2], (width, height)), np.reshape(_N[: ,1], (width, height)), np.reshape(_N[: ,0], (width, height))]))  

    N[np.isnan(N)] = 0



    return N,scount

