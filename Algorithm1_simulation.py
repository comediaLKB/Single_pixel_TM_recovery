
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:09:15 2022
@author: 84355
"""
# text
# %matplotlib qt

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn

n_nonlin = 4
N_mode = 16
N_slm = N_mode**2
N_obj = 5
N_m = 16 * N_obj * N_slm

noise = 0
use_gpu = True

# Check GPU availability
if use_gpu:
    use_gpu = torch.cuda.is_available()
    device = "cuda:0"
else:
    device = "cpu"
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

# define the transmission matrix T
T_gt = 1/N_mode * (torch.randn(N_slm,N_obj).to(device) + 1j*torch.randn(N_slm,N_obj).to(device))

# define measurement matrix A
A = torch.exp(1j * 2 * torch.pi * torch.rand(N_m,N_slm)).to(device)

# define the forward model
def forward(A, T, noise):
    forward_op = torch.matmul(A,T).to(device)
    forward_op = torch.abs(forward_op) ** n_nonlin 
    return (torch.matmul(forward_op,torch.ones(T.size()[1],1).to(device)) + noise*torch.rand(A.size()[0],1).to(device))

# pmt signal collection
Y = forward(A = A, T = T_gt, noise = noise)

#%% Algorithm 1
# guess N_obj
N_obj_guess = N_obj

# grad decend param set
n_iter = 3000    

# Random initialization
T_est = 1/N_mode * (torch.randn(N_slm,N_obj_guess).to(device) + 1j*torch.randn(N_slm,N_obj_guess).to(device))
step =  5e-2

# gradient descent
T_est_grad =  torch.zeros(N_slm,N_obj).cfloat().to(device)
loss_vec = torch.zeros((n_iter,)).to(device)
criterion = nn.MSELoss()
for i_iter in tqdm(range(n_iter)): 
          
    # forward
    Y_est = forward(A = A, T = T_est, noise = noise).to(device)

    # backprop loss
    loss = criterion(Y_est, Y)
    loss_vec[i_iter] = loss
             
    # gradient
    T_est_grad = 4 * torch.matmul( torch.transpose(torch.conj(A),0,1).cfloat().to(device), ((Y_est - Y).cfloat().to(device) * ((torch.abs(torch.matmul(A,T_est))**2).cfloat() * (torch.matmul(A,T_est)).cfloat())))
    
    # update

    T_est -= step * T_est_grad/torch.norm(T_est_grad)  
         

fig, axs = plt.subplots(1,1)
axs.plot(loss_vec.cpu().detach().numpy())
axs.set_yscale('log')

print('loss_mse = ', loss.cpu().detach().numpy())

# check the fedility of the reconstructed T
corel = torch.zeros(N_obj, N_obj_guess).to(device)
for tt in range(N_obj):
    for jj in range(N_obj_guess):
        corel[tt,jj] = torch.abs(torch.matmul(torch.conj(T_gt[:,tt]),T_est[:,jj])) / (torch.linalg.norm(T_gt[:,tt])*torch.linalg.norm(T_est[:,jj]))

cc, idex = torch.max(corel,0)
cc_mean = torch.mean(cc)

print('T_est fedility = ', cc_mean.cpu().detach().numpy())


