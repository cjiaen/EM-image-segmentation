# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:48:42 2017

@author: cjiaen
"""

import os
import numpy as np
from functions.io_data import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

FILEPATH = r'C:\Users\cjiaen\Documents\Sem1\CS5340_UNCERTAINTY\Project\CS5340-Project\a2'

#calculate PDF of multivariate gaussian
def pdf_multivariate_gaussian(x, mean_k, covar):
    x = x.reshape(mean_k.shape)
    #print("Shape of x is {}".format(x.shape))
    #print("Shape of mean is {}".format(mean_k.shape))
    #print("Shape of covar is {}".format(covar.shape))
    
    prob1 = (1/(((2*np.pi)**(3/2.0))*np.sqrt(np.linalg.det(covar))))
    #print("prob1 is {}".format(prob1))
    prob2 = np.exp(-0.5*np.dot(np.dot((x-mean_k).T, np.linalg.inv(covar)),(x-mean_k)))
    #print("prob2 is {}".format(prob2))
    #print("shape of prob2 is {}".format(prob2.shape))
    prob = prob1*prob2
    #print(prob)
    #ref = multivariate_normal.pdf(x,mean_k.reshape(3),covar)
    #assert(prob == ref)
    return(prob)

#implementation of EM algorithm
def EM(data,K,EPOCHS,THRESHOLD):
    print("Initializing...")
    #initialize values
    H,W,C = data.shape
    #reshape data
    data = data.transpose((2,0,1)).reshape(3,-1) #C,H*W
    #mixing coefficient
    pi_k = np.random.dirichlet(np.random.randn(K), size=1).reshape(K) #background, foreground
    #indicator variable
    z_nk = np.random.randint(2, size=(H*W,K), dtype=int) #H*W, K
    z_nk[:,1] = 1 - z_nk[:,0]
    #responsibility variable
    responsibility = np.zeros(z_nk.shape) #H*W, K
    responsibility_temp = np.zeros(z_nk.shape)
    #Gaussian covariates per channel per cluster
    covar = [np.eye(C)]*K #K list of matrices of shape C,C
    #Gaussian mean per channel per cluster (choose 2 random data points)
    mean_k = [data[:,np.random.choice(data.shape[1])] for clusters in range(K)]
    log_likelihood = []
    
    for epoch in range(EPOCHS):
        #check for convergence of loglikelihood
        if len(log_likelihood) > 1 and (log_likelihood[-1] - log_likelihood[-2] < THRESHOLD):
            break
        else:
            #evaluate responsibilities
            for cluster in range(K):    
                responsibility_temp[:,cluster] = pi_k[cluster] * np.apply_along_axis(pdf_multivariate_gaussian, 0, data, mean_k[cluster], covar[cluster])
            responsibility = np.divide(responsibility_temp,np.sum(responsibility_temp, axis=1,keepdims=True))
            #if nan, assume uniform distribution
            responsibility[np.isnan(responsibility)] = 1.0/K
            N_k = np.sum(responsibility, axis=0)
            
            #update mean and covariance
            for cluster in range(K):
                #update mean
                mean_k[cluster] = (1/N_k[cluster])*np.sum((responsibility[:,cluster]*data),axis = 1)
                mean_k[cluster] = mean_k[cluster].reshape(3,-1)
                #update covariance
                covar[cluster] = (1/N_k[cluster])*np.dot(responsibility[:,cluster]*np.subtract(data,mean_k[cluster]), np.subtract(data,mean_k[cluster]).T)
            
            #update mixture coefficients
            pi_k = N_k/(H*W)
            
            #update log-likelihood
            for cluster in range(K):
                responsibility_temp[:,cluster] = pi_k[cluster] * np.apply_along_axis(pdf_multivariate_gaussian, 0, data, mean_k[cluster], covar[cluster])
            log_likelihood.append(sum(np.log(np.sum(responsibility_temp,axis = 1))))
        print("Epoch {}: Log likelihood = {}".format(epoch+1, log_likelihood[-1]))
    return(responsibility)

def get_image(R, image, filename):    
    #background
    mask = R[:,:,0]
    foreground = copy.deepcopy(image)
    foreground[mask > 0.5] = 0
    img = Image.fromarray(foreground, 'RGB')
    img.save(filename+'_foreground.png')
    #img.show()
    
    mask_img = copy.deepcopy(image)
    mask_img[mask > 0.5] = 0
    mask_img[mask <= 0.5] = 255
    img = Image.fromarray(mask_img, 'RGB')
    img.save(filename+'_mask.png')
    #img.show()
    
    mask = R[:,:,1]
    background = copy.deepcopy(image)
    background[mask >= 0.5] = 0
    img = Image.fromarray(background, 'RGB')
    img.save(filename+'_background.png')
    #img.show()
    
    return 0

#load image
image1 = np.array(Image.open(os.path.join(FILEPATH,'cow.jpg')))
image2 = np.array(Image.open(os.path.join(FILEPATH,'fox.jpg')))
image3 = np.array(Image.open(os.path.join(FILEPATH,'owl.jpg')))
image4 = np.array(Image.open(os.path.join(FILEPATH,'zebra.jpg')))
#image_cow = read_data(os.path.join(FILEPATH,'cow.jpg'), True, True)
data1 = image1
data2 = image2
data3 = image3
data4 = image4
K = 2 #clusters

R1 = EM(data1,K,15,10)
R1 = R1.reshape((data1.shape[0], data1.shape[1], K))
get_image(R1, image1, "cow")

R2 = EM(data2,K,15,10)
R2 = R2.reshape((data2.shape[0], data2.shape[1], K))
get_image(R2, image2, "fox")

####get image for owl#####
R3 = EM(data3,3,15,10)
R3 = R3.reshape((data3.shape[0], data3.shape[1], 3))
#background
mask = R3[:,:,1]
background = copy.deepcopy(image3)
background[mask >= 0.5] = 0
img = Image.fromarray(background, 'RGB')
img.show()
img.save('owl_background.png')

mask = R3[:,:,1]
foreground = copy.deepcopy(image3)
foreground[mask <= 0.5] = 0
img = Image.fromarray(foreground, 'RGB')
img.show()
img.save('owl_foreground.png')

mask_img = copy.deepcopy(image3)
mask_img[mask > 0.5] = 0
mask_img[mask <= 0.5] = 255
img = Image.fromarray(mask_img, 'RGB')
img.save('owl_mask.png')
img.show()
################################################
####get image for zebra#####
R4 = EM(data4,3,15,10)
R4 = R4.reshape((data4.shape[0], data4.shape[1], 3))

#background
mask0 = R4[:,:,0]
background = copy.deepcopy(image4)
background[mask0 >= 0.5] = 0
mask2 = R4[:,:,2]
background[mask2 >= 0.5] = 0
img = Image.fromarray(background, 'RGB')
img.show()
img.save('zebra_background4.png')

mask = R4[:,:,0]
foreground = copy.deepcopy(image4)
foreground[mask <= 0.5] = 0
img = Image.fromarray(foreground, 'RGB')
img.show()
img.save('zebra_foreground4.png')

mask_img = copy.deepcopy(image4)
#mask_img = 255
comb_mask = np.zeros(mask0.shape)
comb_mask[((mask0 >= 0.5)|(mask2 >= 0.5))] = True
mask_img[comb_mask == True] = 0
mask_img[comb_mask == False] = 255
mask_img[mask2 >= 0.5] = 0
img = Image.fromarray(mask_img, 'RGB')
img.show()

img.save('zebra_mask4.png')
