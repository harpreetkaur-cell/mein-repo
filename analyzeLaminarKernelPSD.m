% PSD of the kernel part of the laminar data
%   q_CFD = q_BI + q_kernel,  q_BI = h_Gauss * u
%
% Figure 1: PSD of q from the data, of the Bayesian model prediction q_BI
%           and of the kernel part q_kernel = q - q_BI
% Figure 2: AC and PSD of q_kernel with two fitted kernels (cf. Zimmermann Fig. 6)

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

nWelch = 2^15;       % Welch segment length -> df = fs/nWelch ~ 30 Hz

%% Gaussian pulse model (cached, see analyzeLaminarSpectra)
if isempty(config.inference.dataVariance)
    ceTag = '_CeInferred';
else
    ceTag = '';
end
cacheFile = sprintf('local/laminarPosterior_%s%g%s.mat', config.preproc.DSmode, config.preproc.DSvalue, ceTag);
load(cacheFile, 'posterior');
nPulse = numel(posterior.a)/3;

%% Step 1: q_BI and q_kernel in the time domain
[~, ~, ~, t_max] = generatePrior(nPulse, T_c, config);
t_h = (0:ceil(t_max*T_c*fs))'/fs;
hBI = calculateImpulseResponse(posterior.b, t_h./T_c, T_c);
qBI = fftfilt(hBI/fs, u_full);          % q_BI = h_Gauss * u

valid = numel(t_h):numel(u_full);       % skip the start-up of the convolution
q  = q_full(valid);
qBI = qBI(valid);
qK = q - qBI;                           % q_kernel

%% PSDs (Welch, 4-term Blackman-Harris, 66.1% overlap, 95% CI)
win = blackmanharris(nWelch);
nOverlap = round(0.661*nWelch);
[Pqq, f, PqqCI] = pwelch(q,   win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);
[Pbb, ~, PbbCI] = pwelch(qBI, win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);
[Pkk, ~, PkkCI] = pwelch(qK,  win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);

% cutoff frequency of the forcing signal (99.9% of its energy)
Puu = pwelch(u_full, win, nOverlap, nWelch, fs);
f_cut = f(find(cumsum(Puu)/sum(Puu) >= 0.999, 1, 'first'));

%% Step 2: kernel fitted to q_kernel (based on Zimmermann et al.)
% Two kernels:
%   1: Matern(3/2) x cos                  -> low-pass with one peak (Zimmermann Eq. 17)
%      parameters sigma^2, l, f_c
%   2: Matern(3/2) + Matern(3/2) x cos    -> broad low-pass plus a separate narrow peak
%      parameters sigma_1^2, l_1 (low-pass), sigma_2^2, l_2, f_c (peak)
% Both kernels are multiplied by a low-pass at the cutoff frequency of the forcing,
% 1/(1 + (f/f_cut)^(2n)), because q_kernel = q - h*u has no energy where u has none.
% The order n is fitted.
% A high-pass (Zimmermann Eq. 19) is left out: the PSD of q_kernel is flat at low f.
% Fitted with the Whittle likelihood (Zimmermann Eq. 20) up to fFitMax.
modelNames = {'Matern x cos x LP', 'Matern + Matern x cos, x LP'};
fFitMax = 3000;                         % just above f_cut, below the numerical noise floor
fitBins = f > 0 & f <= fFitMax;
opts = optimset('Display', 'off', 'MaxFunEvals', 4e4, 'MaxIter', 4e4, 'TolX', 1e-8, 'TolFun', 1e-8);
starts = {log([var(qK); T_c/3; 400; 10]), ...                    % model 1: [sigma^2; l; f_c; n]
          log([repmat([var(qK)/2; T_c/6; var(qK)/2], 1, 3); [0.5 1 2]*T_c; 420 420 420; 10 10 10])}; % model 2: [sigma_1^2; l_1; sigma_2^2; l_2; f_c; n]
thK = cell(1, 2);
for model = 1:2
    whittle = @(th) sum(log(kernelPSD(f(fitBins), th, model, f_cut)) ...
        + Pkk(fitBins)./kernelPSD(f(fitBins), th, model, f_cut));
    th0 = starts{model};
    best = inf;
    for j = 1:size(th0, 2)              % a few starts for the peak width
        [th, val] = fminsearch(whittle, th0(:,j), opts);
        if val < best, best = val; thK{model} = th; end
    end
end

% autocorrelation of q_kernel and of the kernels (inverse Fourier transform of the PSD)
lagMax = 3;                             % in units of T_c
nLag = round(lagMax*T_c*fs);
[acK, lags] = xcorr(qK, nLag, 'coeff');
acK = acK(lags >= 0);
tLag = (0:nLag)'/fs;
tAC = linspace(0, lagMax*T_c, 300)';
fInt = (0:1:5e4)';                      % frequency grid for the inverse transform
acModel = zeros(numel(tAC), 2);
for model = 1:2
    Pint = kernelPSD(fInt, thK{model}, model, f_cut);
    acModel(:,model) = cos(2*pi*tAC*fInt') * Pint / sum(Pint);
end

%% Plots
red   = [0.85 0.10 0.10];
blue  = [0.10 0.10 0.85];
grey  = [0.75 0.75 0.75];
fLim = [20 1e4];
ok = f > 0;

figure(1); clf
patchCI(f(ok), PqqCI(ok,:), grey); hold on
patchCI(f(ok), PbbCI(ok,:), red);
patchCI(f(ok), PkkCI(ok,:), blue);
p1 = loglog(f(ok), Pqq(ok), 'k', 'lineWidth', 1);
p2 = loglog(f(ok), Pbb(ok), 'color', red,  'lineWidth', 1.2);
p3 = loglog(f(ok), Pkk(ok), 'color', blue, 'lineWidth', 1.2);
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-18 1e-4]); grid on
legend([p1 p2 p3], 'q_{CFD} (data)', sprintf('q_{BI} (%d Gaussian pulses)', nPulse), ...
    'q_{kernel} = q_{CFD} - q_{BI}', 'location', 'southwest')

% Kernel fits: AC (top) and PSD (bottom), cf. Zimmermann Fig. 6
figure(2); clf
for model = 1:2
    p = exp(thK{model});
    if model == 1
        parText = sprintf('\\sigma^2 = %.1e, l = %.2f T_c, f_c = %.0f Hz, n = %.1f', p(1), p(2)/T_c, p(3), p(4));
    else
        parText = sprintf('\\sigma_1^2 = %.1e, l_1 = %.2f T_c, \\sigma_2^2 = %.1e, l_2 = %.2f T_c, f_c = %.0f Hz, n = %.1f', ...
            p(1), p(2)/T_c, p(3), p(4)/T_c, p(5), p(6));
    end

    subplot(2, 2, model)
    plot(tLag/T_c, acK, 'color', blue, 'lineWidth', 1.2); hold on
    plot(tAC/T_c, acModel(:,model), 'color', red, 'lineWidth', 1.5);
    yline(0, 'k:');
    xlabel('\tau/T_c [-]'); ylabel('AC [-]'); xlim([0 lagMax]); ylim([-0.6 1]); grid on
    title(modelNames{model})

    subplot(2, 2, model + 2)
    patchCI(f(ok), PkkCI(ok,:), blue); hold on
    p1 = loglog(f(ok), Pkk(ok), 'color', blue, 'lineWidth', 1.2);
    p2 = loglog(f(ok), kernelPSD(f(ok), thK{model}, model, f_cut), 'color', red, 'lineWidth', 1.5);
    xline(f_cut, 'k--');
    set(gca, 'xScale', 'log', 'yScale', 'log');
    xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-14 1e-7]); grid on
    legend([p1 p2], 'q_{kernel}', parText, 'location', 'southwest', 'fontSize', 7)
end

function P = kernelPSD(f, th, model, fLP)
    % One-sided PSD [1/Hz] of the kernel, times a low-pass of order n at fLP.
    %   model 1: th = log [sigma^2; l; f_c; n]
    %   model 2: th = log [sigma_1^2; l_1; sigma_2^2; l_2; f_c; n]
    p = exp(th);
    if model == 1
        P = maternCosPSD(f, p(1), p(2), p(3));
    else
        P = maternCosPSD(f, p(1), p(2), 0) + maternCosPSD(f, p(3), p(4), p(5));
    end
    P = P ./ (1 + (f/fLP).^(2*p(end)));         % low-pass at the cutoff frequency of u
end

function P = maternCosPSD(f, s2, l, fc)
    % One-sided PSD [1/Hz] of s2*(1 + sqrt(3)|tau|/l)*exp(-sqrt(3)|tau|/l)*cos(2*pi*fc*tau).
    lam = sqrt(3)/l;
    S = @(w) 4*lam^3 ./ (lam^2 + w.^2).^2;      % Matern(3/2) with unit variance
    w  = 2*pi*f;
    wc = 2*pi*fc;
    P = 2*s2*0.5*(S(w - wc) + S(w + wc));       % x cos(omega_c tau) shifts the spectrum
end

function patchCI(x, ci, colour)
    % Shaded confidence band between the two columns of ci.
    fill([x; flipud(x)], [ci(:,1); flipud(ci(:,2))], colour, 'faceAlpha', 0.3, 'edgeColor', 'none');
end
