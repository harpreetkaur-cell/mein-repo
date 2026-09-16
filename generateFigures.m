% Generate figures for the laminar flame impulse response analysis.
%
% Before running, ensure that the src, utils and data folders are added to
% your path.

%% !! Run this first !!
% Set paths, system properties and load the full dataset
ptd  = 'data/Laminar/';

% System properties:
L_ref = 4.5e-3; % Flame length
V_ref = 1; % Bulk flow velocity
T_c  = L_ref/V_ref; % Reference time scale for non-dimensionalization

% Load data
rawQ = readmatrix(sprintf('%sheatRelease.csv',ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv',ptd));

t_full = rawU(:,1);
u_full = rawU(:,2);
q_full = rawQ(:,2);
dt = t_full(2) - t_full(1);
t_full_ms = t_full*1e3;

u_full = (u_full - mean(u_full))/mean(u_full);
q_full = (q_full - mean(q_full))/mean(q_full);

%% Figure 2

% Generate default prior, as described in §4.1
config = loadDefaultConfig();
[mu_b,Cb] = generatePrior(1,1,config);

% Prepare plot
f2 = figure(2);
set(f2,'Units','normalized','Position',[0.1 0.1 0.6 0.3])
tiledlayout(1,3,'padding','compact','TileSpacing','compact');

% Plot prior over n
mu_n = mu_b(1);
sig_n = sqrt(Cb(1,1));
n = linspace(-5,5,200);
pn = normpdf(n,mu_n,sig_n);

nexttile(1)
plot(n,pn,'color',getColours(1),'lineWidth',1)
xlabel('$n_i$',interpreter='latex')
ylabel('$p(n_i)$',interpreter='latex')
title('(a) Prior over $n$',interpreter='latex')

% Plot prior over alpha
gam = linspace(-10,log(6),300);
mu_gam = mu_b(2);
sig_gam = sqrt(Cb(2,2));
pgam = normpdf(gam,mu_gam,sig_gam);
alpha = exp(gam);

nexttile(2)
plot(alpha, pgam,'color',getColours(1),'lineWidth',1)
xlabel('${\alpha}_i$', interpreter='latex')
ylabel('$p({\alpha}_i)$', interpreter='latex')
title('(b) Prior over $\alpha$', interpreter='latex')

% Plot the prior over sigma/T_c
bet = linspace(-10,log(1),300);
mu_beta = mu_b(3);
sig_beta = sqrt(Cb(3,3));
pbet = normpdf(bet,mu_beta,sig_beta);
s = exp(bet);

nexttile(3)
plot(s, pbet,'color',getColours(1),'lineWidth',1)
xlabel('$\tilde{\sigma}_i$', interpreter='latex')
ylabel('$p(\tilde{\sigma}_i)$', interpreter='latex')
title('(c) Prior over $\tilde{\sigma}$', interpreter='latex')

%% Figure 4

% Prepare the figure
f5 = figure(4);
set(f5,'Units','normalized','Position',[0.1 0.1 0.5 0.4])

% Plot the normalized fluctuations
hu = plot(t_full_ms,u_full,'color',getColours(1),'lineWidth',1);
hold on
hq = plot(t_full_ms,q_full,'color',getColours(2),'lineWidth',1);
xlim([100, 130]);
grid on
box on
xlabel('time [ms]')
ylabel('normalized fluctuation')
legend([hu,hq],{'$u''/ \bar{u}$','$q''/ \bar{q}$'},'interpreter','latex','box','off');

%% Figure 12: Bayesian inference vs. TFDSI (kernel-based SI) on Laminar data
%
% Comparison of the two impulse-response estimation methods on the
% Laminar data:
%   - Bayesian inference (evidence-based model order selection, LFL free)
%   - TFDSI: regularized FIR estimate via impulseest (kernel-based SI)
%
% The TFDSI FIR horizon is chosen to cover the same 15 ms window as the
% Bayesian impulse response, rather than an arbitrary fixed value.
% See compareTFDSI.m for a standalone version of this comparison.

% General settings
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;
config.inference.dataVariance = 3e-6; 

% Bayesian impulse response
[hBay, ~] = inferImpulseResponse(u_full, q_full, t_full, T_c, modelOrders=1:5, config=config);

hBayms = hBay; hBayms.time = hBayms.time*1e3;
hBayms.val = hBayms.val/1e3;
hBayms.var = hBayms.var/1e6; %skalierungen

% TFDSI impulse response
T_h_compare = 0.015; % impulse response window to compare over [s]
omega = 2*pi*linspace(0,500);

signals = prepareSignals(u_full,q_full,1/dt,0,config);
uds = signals.time_domain.coarse.u;
qds = signals.time_domain.coarse.q;
tds = signals.time_domain.coarse.t; %downsampling werte
dtds = mean(diff(tds));
N_TFDSI = fix((T_h_compare + 2*dtds)/dtds);

[hSI,~] = TFDSI(uds,qds,tds,N_TFDSI,omega);

% Plot
f12 = figure(12);
set(f12,'Units','normalized','Position',[0.1 0.1 0.5 0.4])
hold on
hSIplot  = errorpatch(hSI.time*1e3,hSI.val/1e3,2*hSI.std/1e3,2*hSI.std/1e3,'color',getColours(3));
hBayplot = errorpatch(hBayms.time,hBayms.val,2*sqrt(hBayms.var),2*sqrt(hBayms.var),'color',getColours(1));
xlabel('time [ms]')
ylabel('$h\times 10^{-3}$','interpreter','latex')
legend([hBayplot,hSIplot],{'Bayesian inference','TFDSI (kernel-based SI)'},'box','off');
xlim([0,15]);
grid on
box on
title('Bayesian inference vs. TFDSI on Laminar data')
