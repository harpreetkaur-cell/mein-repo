% PSD analysis of the laminar data
%
% Figure 1: PSD of the forcing signal u and the forced response q, with the
%           cutoff frequency of u
% Figure 2: PSD of q from the data and from the Bayesian model

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
modelOrders = 1:5;

nWelch = 2^15;       % Welch segment length -> df = fs/nWelch ~ 30 Hz
energyFrac = 0.999;  % energy fraction for the cutoff frequency

%% PSD of u and q, cutoff frequency
win = blackmanharris(nWelch);
nOverlap = round(0.661*nWelch);
[Puu, f, PuuCI] = pwelch(u_full, win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);
[Pqq, ~, PqqCI] = pwelch(q_full, win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);

cumU = cumsum(Puu)/sum(Puu);
f_cut = f(find(cumU >= energyFrac, 1, 'first'));

% Nyquist frequency of the down-sampled signals seen by the inference
signalsDS = prepareSignals(u_full, q_full, fs, 0, config);
f_nyq = signalsDS.time_domain.coarse.fs/2;

%% Bayesian inference and model PSD
if isempty(config.inference.dataVariance)
    ceTag = '_CeInferred';
else
    ceTag = '';
end
cacheFile = sprintf('local/laminarPosterior_%s%g%s.mat', config.preproc.DSmode, config.preproc.DSvalue, ceTag);
if isfile(cacheFile)
    load(cacheFile, 'h', 'posterior', 'modelRanking');
else
    [h, posterior, modelRanking] = inferImpulseResponse(u_full, q_full, t_full, T_c, ...
        modelOrders=modelOrders, config=config);
    if ~isfolder('local'); mkdir('local'); end
    save(cacheFile, 'h', 'posterior', 'modelRanking');
end
a = posterior.a;
nPulse = numel(a)/3;

% Model prediction q_BI = h_Gauss * u in the time domain (causal h, as in the inference)
[~, ~, ~, t_max] = generatePrior(nPulse, T_c, config);
t_h = (0:ceil(t_max*T_c*fs))'/fs;
hBI = calculateImpulseResponse(posterior.b, t_h./T_c, T_c);
qBI = fftfilt(hBI/fs, u_full);
qBI = qBI(numel(t_h):end);              % skip the start-up of the convolution
[Pmod, ~, PmodCI] = pwelch(qBI, win, nOverlap, nWelch, fs, 'ConfidenceLevel', 0.95);

%% Plots
red  = [0.85 0.10 0.10];
blue = [0.10 0.10 0.85];
grey = [0.75 0.75 0.75];
fLim = [20 1e4];
ok = f > 0;

% PSD of forcing signal and forced response 
figure(1); clf
patchCI(f(ok), PuuCI(ok,:), blue); hold on
patchCI(f(ok), PqqCI(ok,:), red);
p1 = loglog(f(ok), Puu(ok), 'color', blue, 'lineWidth', 1.2);
p2 = loglog(f(ok), Pqq(ok), 'color', red,  'lineWidth', 1.2);
xline(f_cut, 'k--', sprintf('f_{cut} = %.0f Hz', f_cut), 'labelVerticalAlignment', 'bottom');
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); grid on
legend([p1 p2], 'Forcing signal u', 'Forced response q', 'location', 'southwest')

% PSD of q: data vs. Bayesian model 
figure(2); clf
patchCI(f(ok), PqqCI(ok,:), grey); hold on
patchCI(f(ok), PmodCI(ok,:), red);
p1 = loglog(f(ok), Pqq(ok), 'k', 'lineWidth', 1);
p2 = loglog(f(ok), Pmod(ok), 'color', red, 'lineWidth', 1.5);
xline(f_cut, 'k--', sprintf('f_{cut} = %.0f Hz', f_cut), 'labelVerticalAlignment', 'bottom');
xline(f_nyq, 'k-.', sprintf('f_{max} = %.0f Hz', f_nyq), 'labelVerticalAlignment', 'bottom');
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim(fLim); ylim([1e-18 1e-4]); grid on
legend([p1 p2], 'Data', sprintf('Bayesian model (%d pulses)', nPulse), 'location', 'southwest')

function patchCI(x, ci, colour)
    % Shaded confidence band between the two columns of ci.
    fill([x; flipud(x)], [ci(:,1); flipud(ci(:,2))], colour, 'faceAlpha', 0.3, 'edgeColor', 'none');
end
