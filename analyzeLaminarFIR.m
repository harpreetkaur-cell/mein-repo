% Bayesian FIR reference for the model error of the Gaussian pulse model
%
% Figure 1: evidence over the FIR length
% Figure 2: impulse response, FIR vs. Gaussian pulse model
% Figure 3: FTF gain and phase, FIR vs. Gaussian pulse model

%% Load data
ptd = 'data/Laminar/';
T_c = 4.5e-3;

rawQ = readmatrix(sprintf('%sheatRelease.csv',ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv',ptd));
t_full = rawU(:,1);
u_full = (rawU(:,2) - mean(rawU(:,2)))/mean(rawU(:,2));
q_full = (rawQ(:,2) - mean(rawQ(:,2)))/mean(rawQ(:,2));

fs = 1/mean(diff(t_full));

%% Settings
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';     
config.preproc.DSvalue = 3e-4;
config.inference.dataVariance = [];   % [] = infer Ce by maximizing the marginal likelihood, or fix it, e.g. 1e-5

T_IR = (1:0.25:5)*T_c;  % candidate FIR lengths
nSamples = 5000;        % posterior samples for the FTF credible intervals
rng(1);

%% Gaussian pulse model (cached)
if isempty(config.inference.dataVariance)
    ceTag = '_CeInferred';
else
    ceTag = '';
end
cacheFile = sprintf('local/laminarPosterior_%s%g%s.mat', config.preproc.DSmode, config.preproc.DSvalue, ceTag);
load(cacheFile, 'h', 'posterior');
hDTD = h;

%% Regression matrix on the down-sampled data
signals = prepareSignals(u_full, q_full, fs, max(T_IR), config);
s = signals.time_domain.coarse;
nMax = numel(s.t_h);
k = (nMax:s.n)';        % common valid region, so the evidence is comparable across lengths
X = zeros(numel(k), nMax);
for i = 1:nMax
    X(:,i) = s.u(k-i+1) .* s.dt;
end
y = s.q(k);

%% Evidence over the FIR length
nCoef = min(round(T_IR/s.dt) + 1, nMax);
logML = nan(size(T_IR));
theta = nan(2, numel(T_IR));
theta0 = log([max(abs(hDTD.val)); 0.1*std(y)]);  % log [sp; se]
opts = optimset('Display', 'off', 'TolX', 1e-8, 'TolFun', 1e-8, 'MaxFunEvals', 4000);
for j = 1:numel(T_IR)
    Xj = X(:,1:nCoef(j));
    [theta(:,j), nlml] = fminsearch(@(th) firEvidence(th, Xj, y), theta0, opts);
    logML(j) = -nlml;
end
[~, iBest] = max(logML);

%% Posterior of the most probable FIR
N = nCoef(iBest);
[~, m, C] = firEvidence(theta(:,iBest), X(:,1:N), y);
tFIR = (0:N-1)'*s.dt;
hStd = sqrt(diag(C));

%% FTF of both models up to the Nyquist frequency of the down-sampled data
f = linspace(0, s.fs/2, 400)';
E = exp(-1i*2*pi*f*tFIR') .* s.dt;
H = E*m;
Hs = E*(m + chol(C, 'lower')*randn(N, nSamples));

FIR.gain = abs(H);
FIR.gain95lo = FIR.gain - quantile(abs(Hs), 0.025, 2);
FIR.gain95hi = quantile(abs(Hs), 0.975, 2) - FIR.gain;
FIR.phase = unwrap(angle(H));
dphase = angle(Hs./H);
FIR.phase95lo = -quantile(dphase, 0.025, 2);
FIR.phase95hi =  quantile(dphase, 0.975, 2);

DTD = calculateFTF(posterior.a, 2*pi*f, posterior.Ca);

%% Plots
red  = [0.85 0.10 0.10];
blue = [0.10 0.10 0.85];
nPulse = numel(posterior.a)/3;

% Evidence over the FIR length (cf. Zimmermann Fig. 9)
figure(1); clf
plot(T_IR/T_c, logML - max(logML), 'o-', 'color', blue, 'lineWidth', 1.2); hold on
plot(T_IR(iBest)/T_c, 0, 'o', 'color', red, 'markerFaceColor', red);
xlabel('T_{IR}/T_c [-]'); ylabel('log evidence (rel. to max) [-]'); grid on

% Impulse response
figure(2); clf
p1 = errorpatch(tFIR/T_c, m, 1.96*hStd, 1.96*hStd, 'color', blue); hold on
p2 = errorpatch(hDTD.time/T_c, hDTD.val, 1.96*sqrt(hDTD.var), 1.96*sqrt(hDTD.var), 'color', red);
yline(0, 'k:');
xlabel('t/T_c [-]'); ylabel('h [s^{-1}]'); grid on
legend([p1 p2], sprintf('FIR (T_{IR} = %.2f T_c)', T_IR(iBest)/T_c), ...
    sprintf('Gaussian pulses (%d)', nPulse), 'location', 'northeast')

% FTF gain and phase (cf. Zimmermann Fig. 10)
figure(3); clf
subplot(2,1,1)
% lower bounds kept positive for the log axis
p1 = errorpatch(f, FIR.gain, min(FIR.gain95lo, 0.99*FIR.gain), FIR.gain95hi, 'color', blue); hold on
p2 = errorpatch(f, DTD.gain, min(DTD.gain95lo, 0.99*DTD.gain), DTD.gain95hi, 'color', red);
set(gca, 'yScale', 'log');
ylabel('Gain [-]'); xlim([0 s.fs/2]); ylim([1e-3 3]); grid on
legend([p1 p2], 'FIR', sprintf('Gaussian pulses (%d)', nPulse), 'location', 'northeast')
subplot(2,1,2)
errorpatch(f, FIR.phase, FIR.phase95lo, FIR.phase95hi, 'color', blue); hold on
errorpatch(f, DTD.phase, DTD.phase95lo, DTD.phase95hi, 'color', red);
xlabel('f [Hz]'); ylabel('Phase [rad]'); xlim([0 s.fs/2]); grid on

function [nlml, m, C] = firEvidence(theta, X, y)
    % Negative log evidence of Bayesian linear regression y = X*h + e with
    % h ~ N(0, sp^2 I) and e ~ N(0, se^2 I), theta = log([sp; se]).
    % Also returns the posterior mean m and covariance C of h.
    sp2 = exp(2*theta(1));
    se2 = exp(2*theta(2));
    [n, N] = size(X);
    A = X'*X/se2 + eye(N)/sp2;
    R = chol(A);
    m = R \ (R' \ (X'*y/se2));
    res = y - X*m;
    nlml = 0.5*(n*log(2*pi*se2) + N*log(sp2) + 2*sum(log(diag(R))) ...
        + res'*res/se2 + m'*m/sp2);
    if nargout > 2
        C = R \ (R' \ eye(N));
    end
end
