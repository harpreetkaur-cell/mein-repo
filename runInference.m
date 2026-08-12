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
% Load CSV data

velocityData = readmatrix('data/laminarFlowData/velocityRef1(3).csv');
heatReleaseData = readmatrix('data/laminarFlowData/heatRelease(4).csv');

% Extract signals
t_full = velocityData(:,1);
u_full = velocityData(:,2);
q_full = heatReleaseData(:,2);

% Calculate timestep
dt = mean(diff(t_full));
t_full_ms = t_full*1e3;

% Normalize signals
u_full = (u_full - mean(u_full))/mean(u_full);
q_full = (q_full - mean(q_full))/mean(q_full);

%% System properties (from README)
L_ref = 4.5*10^-3;  % Flame length
V_ref = 1;  % Bulk flow velocity

T_c = L_ref/V_ref;

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
config.inference.dataVariance = 1e-8;
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 300e-6;

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
