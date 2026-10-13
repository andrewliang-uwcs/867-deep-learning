# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.functions import *
from lib.learnable_parameters import *

#Fix seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)

######## Simulation Parameters ########
#Run size: 'smoke' (~1 min, just checks the code runs), 'cpu' (a few hours on a CPU), 'paper' (full settings, GPU needed)
run_mode       = 'cpu'
save_path      = 'results/'                                                 #Directory to save results
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
impaired_flag  = True                                                       #True to include impairments, False otherwise
K              = 64                                                         #Number of antennas
S              = 256                                                        #Number of subcarriers
Delta_f        = torch.tensor(120e3, dtype=torch.float32, device=device)    #Spacing between subcarriers
fc             = torch.tensor(60e9, dtype=torch.float32, device=device)     #Carrier frequency
N0             = torch.tensor(1, dtype=torch.float32, device=device)        #Noise variance at the receivers
SNR_sens_dB    = torch.tensor(15, dtype=torch.float32, device=device)       #Signal-to-noise ratio of sensing
SNR_comm_dB    = torch.tensor(20, dtype=torch.float32, device=device)       #Signal-to-noise ratio of communication
msg_card       = 4                                                          #Comm. constellation size
numTaps        = 5                                                          #Number of taps in the communication4 channel
target_pfa     = 1e-2                                                       #Target false alarm prob. for ISAC
delta_pfa      = {'smoke': 5e-3, 'cpu': 1e-3, 'paper': 1e-4}[run_mode]  #Max deviation from target_pfa (paper: 1e-4)
range_min_glob = torch.tensor(0, dtype=torch.float32, device=device)        #Minimum considered range of the target
range_max_glob = torch.tensor(200, dtype=torch.float32, device=device)      #Maximum considered range of the target
#Antenna spacing
'''
# Generate ordered antenna impairment, such that the antenna positions d*torch.arange(-(K-1)/2.0,(K-1)/2.0+1) are ordered
std_d = lamb/25  #Antenna displacement standard deviation
ant_d = generateImpairments(std_d, lamb, K)
#We should always check that the final positions are ordered
'''
#Here is an example of impairments that produce ordered positions
ant_d = torch.tensor([[0.0030255569, 0.0027658057, 0.0027241893, 0.0027152568, 0.0026999826,
            0.0026906996, 0.0026126332, 0.0025860183, 0.0025710680, 0.0025635406,
            0.0025424187, 0.0025299306, 0.0025263054, 0.0025066731, 0.0024867542,
            0.0024508000, 0.0024486321, 0.0024324458, 0.0024288078, 0.0024131122,
            0.0023973403, 0.0023669580, 0.0023466886, 0.0023346620, 0.0023164046,
            0.0023071333, 0.0022835401, 0.0022781249, 0.0022470967, 0.0022099297,
            0.0022072701, 0.0020638802, 0.0020917924, 0.0022089742, 0.0022406091,
            0.0022769766, 0.0022815398, 0.0022839003, 0.0023076974, 0.0023233502,
            0.0023365330, 0.0023507008, 0.0023888862, 0.0024037808, 0.0024274669,
            0.0024292781, 0.0024454184, 0.0024500827, 0.0024605242, 0.0024989846,
            0.0025112331, 0.0025298016, 0.0025336186, 0.0025491153, 0.0025707644,
            0.0025738038, 0.0025952985, 0.0026659546, 0.0026948208, 0.0027010187,
            0.0027233388, 0.0027394753, 0.0027674662, 0.0030494009]], dtype=torch.float32, device = device).T
#We allow for training of random [theta_min, theta_max] and [range_min, range_max]
theta_mean_min = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
span_min_theta = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
span_max_theta = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min = (range_min_glob + range_max_glob)/2.0                      #This fixes the target range sector to [range_min_glob, range_max_glob]
range_mean_max = (range_min_glob + range_max_glob)/2.0
span_min_range = range_max_glob - range_min_glob
span_max_range = range_max_glob - range_min_glob
Ngrid_angle    = 720                #Number of points in the oversampled grid of angles
Ngrid_range    = 200                #Number of points in the oversampled grid of ranges
batch_size     = {'smoke': 32, 'cpu': 256, 'paper': 3000}[run_mode]
#List of epochs to evaluate the models while learning
if run_mode == 'smoke':
    epoch_test_list = [1, 20]
elif run_mode == 'cpu':
    epoch_test_list = [1, 250, 500, 1000, 1500, 2000]
else:
    epoch_test_list = [1, 200, 400, 600, 700, 800, 1000, 1500, 2000, 2500, 3000, 3500,
                       5000, 6500, 8000, 1e4, 1.25e4, 1.5e4, 2e4, 2.5e4, 3e4,
                       3.5e4, 4e4, 4.5e4, 5e4, 5.5e4, 6e4, 6.5e4, 7e4, 7.5e4,
                       8e4, 8.1e4, 8.2e4, 8.3e4, 8.4e4, 8.42e4, 8.5e4]
#Testing values after training
nTestSamples        = {'smoke': 640, 'cpu': int(1e4), 'paper': int(1.2e6)}[run_mode]  #This value will be slightly changed later to be a multiple of batch_size
theta_min_sens_test = torch.tensor(-40*np.pi/180, dtype=torch.float32, device=device)
theta_max_sens_test = torch.tensor(-20*np.pi/180, dtype=torch.float32, device=device)
range_min_sens_test = torch.tensor(0, dtype=torch.float32, device=device)
range_max_sens_test = torch.tensor(200, dtype=torch.float32, device=device)
theta_min_comm_test = torch.tensor(30*np.pi/180, dtype=torch.float32, device=device)
theta_max_comm_test = torch.tensor(50*np.pi/180, dtype=torch.float32, device=device)
#ISAC Trade-off beam parameters
ISAC_grid_pts  = {'smoke': 2, 'cpu': 4, 'paper': 8}[run_mode]  #Number of points for the ISAC trade-off beam
rho = torch.linspace(0,1,ISAC_grid_pts, device=device)
phi = torch.linspace(0, 2*np.pi*(ISAC_grid_pts-1)/ISAC_grid_pts, ISAC_grid_pts, device=device) #Avoid repeating 0 and 2*pi

######## Other Parameters computed from simulation parameters ########
refConst     = MPSK(msg_card, np.pi/4, device=device)   #Refence constellation (PSK)
lamb         = 3e8 / fc                                 #Wavelength
sigma_sens   = torch.sqrt(10**(SNR_sens_dB/10)*N0/K)    #std of the complex channel gain
angle_res    = 2/K                                      #Angle resolution (roughly)
range_res    = 3e8 / (2*S*Delta_f)                      #Range resolution (roughly)
#Communication channel taps
sigma_sq_sum = 10**(SNR_comm_dB/10)*S*N0
#Normalized communication channel taps (exponential pdp), so that the sum of squares is equal to sigma_sq_sum
sigma_vector_comm = torch.cat((torch.exp(-torch.arange(numTaps, device=device)).view(1,-1), \
                               torch.zeros(1,S-numTaps, device=device)), dim = 1)
sigma_vector_comm *= (torch.sqrt(sigma_sq_sum) / torch.norm(sigma_vector_comm))
# Pixels to look around maximum value for OMP (more details in the model-based training file)
pixels_angle = int(angle_res / (np.pi/Ngrid_angle))
pixels_range = int(range_res / ((range_max_glob - range_min_glob)/Ngrid_range))
#Assumed inter-antenna spacing
if impaired_flag:
    assumed_d = lamb/2.0*torch.ones((K,1), dtype=torch.float32, device=device)  #Unknown impairments
else:
    assumed_d = torch.clone(ant_d)                                              #Known impairments
#Round number of test samples to be a multiple of the batch size
numTestIt      = nTestSamples // batch_size                                       #Number of test iterations
nTestSamples   = numTestIt*batch_size                                             #Recompute test samples to be a multiple of batch_Size
thresholds_pfa  = torch.linspace(463, 462, 3, device=device)                      #List of thresholds for sensing testing and a fix Pfa

######## Saving results ########
import os, datetime
run_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')   #Fixed once per run, so all files of a run share it
def result_path(file_name):
    '''Return a unique path results/<file_name>_<run_mode>_<timestamp>.npz, creating the folder if needed.'''
    os.makedirs(save_path, exist_ok=True)
    return os.path.join(save_path, f'{file_name}_{run_mode}_{run_timestamp}.npz')

######## NN-related parameters ########
train_it         = {'smoke': 20, 'cpu': int(2e3), 'paper': int(8.5e4)}[run_mode]  #Number of training iterations
network          = PertNet(assumed_d.cpu()).to(device)
#Define different learning rates and optimizers for the case of sequential training
lr_super         = 4e-7
optimizerSuper   = torch.optim.Adam(list(network.parameters()), lr = lr_super)
scheduler_flag   = True                                                           #Flag to use scheduler if True or skip it if False
gamma_sch        = 0.1
scheduler_super  = torch.optim.lr_scheduler.StepLR(optimizerSuper, step_size=max(1, int(train_it*50000/85000)),   #Paper: 50000 (of 8.5e4 its)
                                                   gamma=gamma_sch)
lr_unsuper       = 5e-7
optimizerUnsuper = torch.optim.Adam(list(network.parameters()), lr = lr_unsuper,
                                    maximize=True)

criterionLoss    = nn.MSELoss()
train_it_super   = train_it
train_it_unsuper = train_it-train_it_super

ratioLabeled     = train_it_super/train_it       #Value in [0,1] indicating how much labeled data to use against unlabelled data

#Flag to indicate whether we test just the model in the first stage of sequential training
test_first_stage = True

#Create an oversampled dictionary of possible target ranges and angles
range_grid = torch.linspace(range_min_glob, range_max_glob, Ngrid_range, device=device)