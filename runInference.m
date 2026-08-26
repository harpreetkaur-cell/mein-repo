% Run Bayesian inference to estimate the flame impulse response posterior
% from the laminar flow input-output data, without generating any plots.
%
% Before running, ensure that the src, utils and data folders are added to
% your path.

%% Set paths and system properties
ptd  = 'data/Laminar/';

L_ref = 4.5e-3; % Flame length
V_ref = 1;      % Bulk flow velocity
T_c  = L_ref/V_ref; % Reference time scale for non-dimensionalization

%% Load data
rawQ = readmatrix(sprintf('%sheatRelease.csv',ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv',ptd));

t_full = rawU(:,1);
u_full = rawU(:,2);
q_full = rawQ(:,2);

u_full = (u_full - mean(u_full))/mean(u_full);
q_full = (q_full - mean(q_full))/mean(q_full);

%% Run Bayesian inference
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;

% Noise variant 1: fix the data variance to a small value close to the
% naturally evidence-maximized estimate (~2.26e-6), instead of estimating
% it via evidence maximization. Values far below this break the Occam
% factor balance in the evidence approximation and give unstable,
% unphysically inflated model orders (see sweep results).
config.inference.dataVariance = 1e-6;

[h, posterior, modelRanking] = inferImpulseResponse(u_full, q_full, t_full, T_c, modelOrders=1:5, config=config);
