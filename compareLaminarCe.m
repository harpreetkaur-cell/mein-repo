% Comparison of the Gaussian pulse model with fixed and inferred Ce
%   fixed:    Ce = 1e-5
%   inferred: Ce from maximizing the marginal likelihood
% Needs both cached posteriors, created by analyzeLaminarSpectra with
% config.inference.dataVariance = 1e-5 and = [].
%
% Figure 1: PSD of q, data vs. Bayesian model (cf. analyzeLaminarSpectra Fig. 2)
% Figure 2: impulse response, FIR vs. Gaussian pulses (cf. analyzeLaminarFIR Fig. 2)
% Figure 3: FTF gain and phase, FIR vs. Gaussian pulses (cf. analyzeLaminarFIR Fig. 3)
% Figure 4: PSD of q_CFD, q_BI and q_kernel = q_CFD - q_BI for the inferred Ce
%           (cf. analyzeLaminarKernelPSD Fig. 1)
% Figure 5: AC and PSD of q_kernel = q - q_BI with the fitted kernel
%           Matern + Matern x cos (cf. analyzeLaminarKernelPSD)

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

ceTags = {'', '_CeInferred'};          % cache names: fixed Ce = 1e-5, inferred Ce
nCase  = numel(ceTags);

nWelch   = 2^15;                       % Welch segment length on the full data
T_IR     = (1:0.25:5)*T_c;             % candidate FIR lengths
nSamples = 5000;                       % posterior samples for the FIR credible intervals
lagMax   = 3;                          % maximum lag of the AC in units of T_c
rng(1);

%% PSD of the data and cutoff frequency
win = blackmanharris(nWelch);
nOverlap = round(0.661*nWelch);
[Puu, f] = pwelch(u_full, win, nOverlap, nWelch, fs);
f_cut = f(find(cumsum(Puu)/sum(Puu) >= 0.999, 1, 'first'));

%% Both Gaussian pulse models: q_BI, q_kernel and kernel fit
opts = optimset('Display', 'off', 'MaxFunEvals', 4e4, 'MaxIter', 4e4, 'TolX', 1e-8, 'TolFun', 1e-8);
fitBins = f > 0 & f <= f_cut;
nLag = round(lagMax*T_c*fs);
tLag = (0:nLag)'/fs;
tAC  = linspace(0, lagMax*T_c, 300)';
fInt = (0:1:5e4)';                     % frequency grid for the inverse transform of the kernel PSD

res = struct([]);
for c = 1:nCase
    cacheFile = sprintf('local/laminarPosterior_%s%g%s.mat', ...
        config.preproc.DSmode, config.preproc.DSvalue, ceTags{c});
    load(cacheFile, 'h', 'posterior');
    nPulse = numel(posterior.a)/3;

    % q_BI = h_Gauss * u (causal, as in the inference) and q_kernel = q - q_BI
    [~, ~, ~, t_max] = generatePrior(nPulse, T_c, config);
    t_h = (0:ceil(t_max*T_c*fs))'/fs;
    hBI = calculateImpulseResponse(posterior.b, t_h./T_c, T_c);
    qBI = fftfilt(hBI/fs, u_full);
    valid = numel(t_h):numel(u_full);
    q   = q_full(valid);
    qBI = qBI(valid);
    qK  = q - qBI;

    [Pqq, ~, PqqCI] = pwelch(q,   win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);
    [Pbb, ~, PbbCI] = pwelch(qBI, win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);
    [Pkk, ~, PkkCI] = pwelch(qK,  win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);

    % kernel Matern + Matern x cos, Whittle fit up to f_cut (cf. analyzeLaminarKernelPSD)
    whittle = @(th) sum(log(sumKernelPSD(f(fitBins), th)) + Pkk(fitBins)./sumKernelPSD(f(fitBins), th));
    th0 = log([repmat([var(qK)/2; T_c/6; var(qK)/2], 1, 3); [0.5 1 2]*T_c; 420 420 420]);
    best = inf;
    for j = 1:size(th0, 2)
        [th, val] = fminsearch(whittle, th0(:,j), opts);
        if val < best, best = val; thK = th; end
    end
    Pint = sumKernelPSD(fInt, thK);
    [acK, lags] = xcorr(qK, nLag, 'coeff');

    res(c).posterior = posterior;
    res(c).h      = h;
    res(c).nPulse = nPulse;
    res(c).Ce     = posterior.Ce;
    res(c).Pqq = Pqq;  res(c).PqqCI = PqqCI;
    res(c).Pbb = Pbb;  res(c).PbbCI = PbbCI;
    res(c).Pkk = Pkk;  res(c).PkkCI = PkkCI;
    res(c).thK   = thK;
    res(c).acK   = acK(lags >= 0);
    res(c).acMod = cos(2*pi*tAC*fInt') * Pint / sum(Pint);
end

%% FIR reference (independent of Ce, cf. analyzeLaminarFIR)
signals = prepareSignals(u_full, q_full, fs, max(T_IR), config);
s = signals.time_domain.coarse;
nMax = numel(s.t_h);
k = (nMax:s.n)';
X = zeros(numel(k), nMax);
for i = 1:nMax
    X(:,i) = s.u(k-i+1) .* s.dt;
end
y = s.q(k);

nCoef = min(round(T_IR/s.dt) + 1, nMax);
logML = nan(size(T_IR));
theta = nan(2, numel(T_IR));
theta0 = log([max(abs(res(1).h.val)); 0.1*std(y)]);
optsFIR = optimset('Display', 'off', 'TolX', 1e-8, 'TolFun', 1e-8, 'MaxFunEvals', 4000);
for j = 1:numel(T_IR)
    [theta(:,j), nlml] = fminsearch(@(th) firEvidence(th, X(:,1:nCoef(j)), y), theta0, optsFIR);
    logML(j) = -nlml;
end
[~, iBest] = max(logML);
N = nCoef(iBest);
[~, m, C] = firEvidence(theta(:,iBest), X(:,1:N), y);
tFIR = (0:N-1)'*s.dt;
hStd = sqrt(diag(C));

fF = linspace(0, s.fs/2, 400)';
E  = exp(-1i*2*pi*fF*tFIR') .* s.dt;
H  = E*m;
Hs = E*(m + chol(C, 'lower')*randn(N, nSamples));
FIR.gain = abs(H);
FIR.gain95lo = FIR.gain - quantile(abs(Hs), 0.025, 2);
FIR.gain95hi = quantile(abs(Hs), 0.975, 2) - FIR.gain;
FIR.phase = unwrap(angle(H));
dphase = angle(Hs./H);
FIR.phase95lo = -quantile(dphase, 0.025, 2);
FIR.phase95hi =  quantile(dphase, 0.975, 2);

for c = 1:nCase
    res(c).DTD = calculateFTF(res(c).posterior.a, 2*pi*fF, res(c).posterior.Ca);
end

%% Plots
blue   = [0.10 0.10 0.85];
colour = {[0.93 0.55 0.10], [0.85 0.10 0.10]};   % fixed Ce: orange, inferred Ce: red
names  = cell(1, nCase);
for c = 1:nCase
    if isempty(ceTags{c})
        names{c} = sprintf('C_e = %.1e fixed, %d pulses', res(c).Ce, res(c).nPulse);
    else
        names{c} = sprintf('C_e = %.1e inferred, %d pulses', res(c).Ce, res(c).nPulse);
    end
end
grey = [0.75 0.75 0.75];
fLim = [20 1e4];
ok = f > 0;

% PSD of q: data vs. Bayesian model
figure(1); clf
patchCI(f(ok), res(1).PqqCI(ok,:), grey); hold on
p = gobjects(1, nCase+1);
p(1) = loglog(f(ok), res(1).Pqq(ok), 'k', 'lineWidth', 1);
for c = 1:nCase
    patchCI(f(ok), res(c).PbbCI(ok,:), colour{c});
    p(c+1) = loglog(f(ok), res(c).Pbb(ok), 'color', colour{c}, 'lineWidth', 1.5);
end
xline(f_cut, 'k--', sprintf('f_{cut} = %.0f Hz', f_cut), 'labelVerticalAlignment', 'bottom');
xline(s.fs/2, 'k-.', sprintf('f_{max} = %.0f Hz', s.fs/2), 'labelVerticalAlignment', 'bottom');
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-18 1e-4]); grid on
legend(p, ['Data', names], 'location', 'southwest')

% Impulse response
figure(2); clf
p = gobjects(1, nCase+1);
p(1) = errorpatch(tFIR/T_c, m, 1.96*hStd, 1.96*hStd, 'color', blue); hold on
for c = 1:nCase
    hc = res(c).h;
    p(c+1) = errorpatch(hc.time/T_c, hc.val, 1.96*sqrt(hc.var), 1.96*sqrt(hc.var), 'color', colour{c});
end
yline(0, 'k:');
xlabel('t/T_c [-]'); ylabel('h [s^{-1}]'); xlim([0 3]); grid on
legend(p, [sprintf('FIR (T_{IR} = %.2f T_c)', T_IR(iBest)/T_c), names], 'location', 'northeast')

% FTF gain and phase
figure(3); clf
subplot(2,1,1)
p = gobjects(1, nCase+1);
p(1) = errorpatch(fF, FIR.gain, min(FIR.gain95lo, 0.99*FIR.gain), FIR.gain95hi, 'color', blue); hold on
for c = 1:nCase
    D = res(c).DTD;
    p(c+1) = errorpatch(fF, D.gain, min(D.gain95lo, 0.99*D.gain), D.gain95hi, 'color', colour{c});
end
set(gca, 'yScale', 'log');
ylabel('Gain [-]'); xlim([0 s.fs/2]); ylim([1e-3 3]); grid on
legend(p, ['FIR', names], 'location', 'northeast')
subplot(2,1,2)
errorpatch(fF, FIR.phase, FIR.phase95lo, FIR.phase95hi, 'color', blue); hold on
for c = 1:nCase
    D = res(c).DTD;
    errorpatch(fF, D.phase, D.phase95lo, D.phase95hi, 'color', colour{c});
end
xlabel('f [Hz]'); ylabel('Phase [rad]'); xlim([0 s.fs/2]); grid on

% q_kernel and fitted kernel Matern + Matern x cos
figure(5); clf
subplot(2,1,1)
for c = 1:nCase
    plot(tLag/T_c, res(c).acK, 'color', colour{c}, 'lineWidth', 1.2); hold on
    plot(tAC/T_c, res(c).acMod, '--', 'color', colour{c}, 'lineWidth', 1.5);
end
yline(0, 'k:');
xlabel('\tau/T_c [-]'); ylabel('AC [-]'); xlim([0 lagMax]); ylim([-0.6 1]); grid on
subplot(2,1,2)
p = gobjects(1, 2*nCase);
labels = cell(1, 2*nCase);
for c = 1:nCase
    pk = exp(res(c).thK);
    patchCI(f(ok), res(c).PkkCI(ok,:), colour{c}); hold on
    p(2*c-1) = loglog(f(ok), res(c).Pkk(ok), 'color', colour{c}, 'lineWidth', 1.2);
    p(2*c)   = loglog(f(ok), sumKernelPSD(f(ok), res(c).thK), '--', 'color', colour{c}, 'lineWidth', 1.5);
    labels{2*c-1} = sprintf('q_{kernel}, %s', names{c});
    labels{2*c}   = sprintf('kernel fit: \\sigma_1^2 = %.1e, l_1 = %.2f T_c, \\sigma_2^2 = %.1e, l_2 = %.2f T_c, f_c = %.0f Hz', ...
        pk(1), pk(2)/T_c, pk(3), pk(4)/T_c, pk(5));
end
xline(f_cut, 'k--');
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-14 1e-7]); grid on
legend(p, labels, 'location', 'southwest')

% q_CFD, q_BI and q_kernel for the inferred Ce
cInf = find(~cellfun(@isempty, ceTags), 1);
r = res(cInf);
figure(4); clf
patchCI(f(ok), r.PqqCI(ok,:), grey); hold on
patchCI(f(ok), r.PbbCI(ok,:), colour{cInf});
patchCI(f(ok), r.PkkCI(ok,:), blue);
p1 = loglog(f(ok), r.Pqq(ok), 'k', 'lineWidth', 1);
p2 = loglog(f(ok), r.Pbb(ok), 'color', colour{cInf}, 'lineWidth', 1.2);
p3 = loglog(f(ok), r.Pkk(ok), 'color', blue, 'lineWidth', 1.2);
xline(f_cut, 'k--', sprintf('f_{cut} = %.0f Hz', f_cut), 'labelVerticalAlignment', 'bottom');
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-18 1e-4]); grid on
legend([p1 p2 p3], 'q_{CFD} (data)', sprintf('q_{BI} (%s)', names{cInf}), ...
    'q_{kernel} = q_{CFD} - q_{BI}', 'location', 'southwest')

function P = sumKernelPSD(f, th)
    % One-sided PSD [1/Hz] of Matern(3/2) + Matern(3/2) x cos,
    % th = log [sigma_1^2; l_1; sigma_2^2; l_2; f_c].
    p = exp(th);
    P = maternCosPSD(f, p(1), p(2), 0) + maternCosPSD(f, p(3), p(4), p(5));
end

function P = maternCosPSD(f, s2, l, fc)
    % One-sided PSD [1/Hz] of s2*(1 + sqrt(3)|tau|/l)*exp(-sqrt(3)|tau|/l)*cos(2*pi*fc*tau).
    lam = sqrt(3)/l;
    S = @(w) 4*lam^3 ./ (lam^2 + w.^2).^2;      % Matern(3/2) with unit variance
    w  = 2*pi*f;
    wc = 2*pi*fc;
    P = 2*s2*0.5*(S(w - wc) + S(w + wc));       % x cos(omega_c tau) shifts the spectrum
end

function [nlml, m, C] = firEvidence(theta, X, y)
    % Negative log evidence of Bayesian linear regression y = X*h + e with
    % h ~ N(0, sp^2 I) and e ~ N(0, se^2 I), theta = log([sp; se]).
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

function patchCI(x, ci, colour)
    % Shaded confidence band between the two columns of ci.
    fill([x; flipud(x)], [ci(:,1); flipud(ci(:,2))], colour, 'faceAlpha', 0.3, 'edgeColor', 'none');
end
