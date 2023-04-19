
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:09:15 2022
@author: 84355
"""

# %matplotlib qt

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import h5py

# load the object pic
with h5py.File('person.h5', "r") as f:
     person = f['person'][()]
    
N = 16
N_os = 10 # how many random patterns we use
N_pat = N_os * N**2

noise = 0 

use_gpu = True

# grad decend param set
n_iter = 4000    
step = 5e-2

# Check GPU availability
if use_gpu:
    use_gpu = torch.cuda.is_available()
    device = "cuda:0"
else:
    device = "cpu"
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

# define the Object
O = torch.from_numpy(np.float32(person/255)).to(device) 
O = torch.reshape(O,(N,N,1))

# padding the signal to prevent the border effect
pad_factor = 2    
pad_side = int((pad_factor - 1) * N/2)
O_pad = torch.nn.functional.pad(O, (0, 0, pad_side, pad_side, pad_side, pad_side), 'constant', 0)

# define the transmission matrix (only one column)

dim = (0, 1)
t_gt = torch.exp(1j*2*torch.pi*torch.rand(N, N, 1)).to(device)
t_gt_pad = torch.fft.ifft2(torch.fft.ifftshift(torch.nn.functional.pad(torch.fft.fftshift(torch.fft.fft2(t_gt, dim=dim), dim=dim), (0,0,pad_side, pad_side, pad_side, pad_side), 'constant', 0),dim=dim),dim=dim)

# measurement

# using GM and SLM to generate A

A = np.zeros((N,N,N_pat),dtype=(complex))
rand_mask_phase = np.random.rand(N, N, N_os)

x = np.linspace(0,1,N)
xm, ym = np.meshgrid(x,x)

for idx in range(N_os):
    for i in range(N):
        for j in range(N):
            kx = i - N/2
            ky = j - N/2
            A[:,:, idx*N**2 + i*N + j] = np.exp(1j * (2*np.pi*rand_mask_phase[:,:,idx] + 2*np.pi*kx*xm + 2*np.pi*ky*ym))
    
A = torch.from_numpy(A).to(device)


A_pad = torch.fft.ifft2(torch.fft.ifftshift(torch.nn.functional.pad(torch.fft.fftshift(torch.fft.fft2(A.cfloat(), dim=dim), dim=dim), (0, 0, pad_side, pad_side, pad_side, pad_side), 'constant', 0),dim=dim),dim=dim)


# define the forward model
def forward(t, A, O, noise):
    
    forward_op = torch.fft.fftshift(torch.fft.fft2(A * t, dim=dim), dim=dim) 
    forward_op = torch.abs(forward_op) ** 4
    return torch.matmul(forward_op.flatten(end_dim=1).T, O.flatten()) + noise*torch.rand(A.size()[2],).to(device)

# pmt signal collection
Y = forward(
    t = t_gt_pad, A = A_pad, O = O_pad, noise = noise
)


t_est = torch.ones(N,N,1).to(device)
O_est = torch.rand(N,N,1).to(device)

snr = 0
fn = 0

O_store = torch.zeros(N,N,10)
psf_store = torch.zeros(N,N,10)


#%% Algorithm 2

while snr<12:
    # scanning
    psf =  torch.fft.fftshift(torch.fft.fft2(torch.conj(t_est) * t_gt, dim=dim))
    O_est = torch.abs(torch.fft.ifftshift( torch.fft.ifft2( torch.fft.fft2(torch.abs(psf)**4,dim=dim) * torch.fft.fft2(O, dim=dim), dim=dim))).to(device)
    O_est = O_est/torch.max(O_est)
    
    O_store[:,:,fn] = torch.squeeze(O_est)
    psf_store[:,:,fn] = torch.squeeze(torch.abs(psf)**2)
    fn = fn + 1

    # Random initialization
    t_est = torch.exp(1j*2*torch.pi * torch.rand(N, N, 1)).to(device)
    t_est.requires_grad = True


    # gradient descent
    loss_vec = torch.zeros((n_iter,)).to(device)

    # -- set optimizer
    criterion = nn.MSELoss()

    for i_iter in tqdm(range(n_iter)): 
  
        # padding
        O_est_pad = torch.nn.functional.pad(O_est, (0, 0, pad_side, pad_side, pad_side, pad_side), 'constant', 0)
        t_est_pad = torch.fft.ifft2(torch.fft.ifftshift(torch.nn.functional.pad(torch.fft.fftshift(torch.fft.fft2(t_est, dim=dim), dim=dim), (0, 0, pad_side, pad_side, pad_side, pad_side), 'constant', 0),dim=dim),dim=dim)

        # forward
        Y_est = forward(
            t = t_est_pad, A = A_pad, O = O_est_pad , noise = 0
        )

        # backprop loss
        loss = criterion(Y_est, Y)
        loss_vec[i_iter] = loss
        loss.backward()
             
        # update
        with torch.no_grad():
             t_est -= step * t_est.grad/torch.norm(t_est.grad)  
             
        # Reset gradients
        t_est.grad.zero_()   
        
    t_est.requires_grad = False   
        
 
    psf =  torch.fft.fftshift(torch.fft.fft2(torch.conj(t_est) * t_gt, dim=dim))
    snr = torch.max(torch.abs(psf))/torch.mean(torch.abs(psf)).to(device)
    print('snr = ', snr.cpu().detach().numpy())



# psf changing in different round
fig = plt.figure(1)
for i in range(fn):
    fig.add_subplot(1,fn,i+1).set_title(str(i))
    plt.imshow(psf_store[:,:,i]**2,'gray')

# O changing in different round
fig = plt.figure(2)
for i in range(fn):
    fig.add_subplot(1,fn,i+1).set_title(str(i))
    plt.imshow(O_store[:,:,i])

