% Generate the figures from Yoko & Polifke, 'Bayesian inference of flame
% impulse responses', CnF [under review], 2026.
% 
% This demonstrates the Bayesian method on input-output data from LES on
% the BRS burner, as reported in Eder et al., '﻿Incompressible versus
% compressible large eddy simulation for the identification of premixed
% flame dynamics', International Journal of Spray and Combustion, 2023.
% 
% Before running, ensure that the src, utils and data folders are added to
% your path.

%% !! Run this first !!
% Set paths, system properties and load the full dataset
ptd  = 'data/BRS_EderSilva23/';

% System properties:
L_ref = 50e-3; % Flame length
V_ref = 11.3; % Bulk flow velocity
T_c  = L_ref/V_ref; % Reference time scale for non-dimensionalization

% Load data
load(sprintf('%sdata_raw_incomp.mat',ptd));
exp_FTF = load(sprintf('%sFTF_exp_30kW_Front_lambda1.3.mat',ptd));

u_full = data_raw_incomp.u;
q_full = data_raw_incomp.y;
dt = data_raw_incomp.Ts;
t_full = (0:length(u_full)-1)*dt;
t_full_ms = t_full*1e3;

u_full = (u_full - mean(u_full))/mean(u_full);
q_full = (q_full - mean(q_full))/mean(q_full);

%% Figure 2
% Check git version is correct
%checkGit();

% Generate default prior, as described in §4.1
config = loadDefaultConfig();
[mu_b,Cb] = generatePrior(1,1,config);

%% Figures 5 - 7    
% Check git version is correct
%checkGit();

% General settings
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;

% -------------------------------------------------------------------------
% Run Bayesian inference for baseline case (LFL free)
% -------------------------------------------------------------------------

[h, posterior] = inferImpulseResponse(u_full, q_full, t_full, T_c, modelOrders=1:5, config=config);

% Unpack posterior struct
a = posterior.a;
Ca = posterior.Ca;

% convert h.time to ms time and reduce its amplitude
hms = h; hms.time = hms.time*1e3;
hms.val = hms.val/1e3;
hms.var = hms.var/1e6;

% Calculate FTF
f = linspace(0,500);
omega = 2*pi*f;
FTF = calculateFTF(a,omega,Ca);

% Pack results for plotting later
baseline.hms = hms;
baseline.FTF = FTF;
baseline.a = a;
baseline.Ca = Ca;

% -------------------------------------------------------------------------
% Run Bayesian inference for LFL = 1
% -------------------------------------------------------------------------

% Update settings
config.model.LFL = 1; % Set low-frequency limit to 1

% Run Bayesian inference
%[h, posterior, modelRanking] = inferImpulseResponse(u_full, q_full, t_full, T_c, modelOrders=1:5, config=config);

% Unpack posterior struct
a = posterior.a;
Ca = posterior.Ca;

% convert h.time to ms and reduce its amplitude
hms = h; hms.time = hms.time*1e3;
hms.val = hms.val/1e3;
hms.var = hms.var/1e6;

FTF = calculateFTF(a,omega,Ca);

% -------------------------------------------------------------------------
% Run SI using the settings from Eder et al. 2023
% -------------------------------------------------------------------------

signals = prepareSignals(u_full,q_full,1/dt,0,config);
uds = signals.time_domain.coarse.u;
qds = signals.time_domain.coarse.q;
tds = signals.time_domain.coarse.t;
N = 39;
[hSI,FTFSI] = TFDSI(uds,qds,tds,N,omega);
