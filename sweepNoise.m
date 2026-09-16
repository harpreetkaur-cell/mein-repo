% Sweep over fixed values of the data-noise variance (Ce) to determine
% which value best explains the laminar flow input-output data.
%
% For each candidate noise variance, config.inference.dataVariance is
% fixed and config.inference.includeNoise is set to true, so Ce is held
% constant throughout the optimization (see estimatePosterior.m) instead
% of being inferred by evidence maximization or removed from the cost.
% The log marginal likelihood (logML) returned by inferImpulseResponse is
% used as the goodness-of-fit metric: the noise value that maximizes
% logML best explains the data.
%
% For reference, the sweep also includes:
%   - the noise-free baseline (includeNoise = false), matching the
%     current setting in runInference.m
%   - the evidence-maximizing (MML) noise estimate (dataVariance = [],
%     includeNoise = true), i.e. letting the optimizer infer Ce itself
%
% Before running, ensure that the src, utils and data folders are added
% to your path.

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

modelOrder = 3; % Fixed model order so results are comparable across noise values

%% Sweep over fixed noise variance values
% Refined sweep: broad sweep (logspace(-8,-2,13)) found the optimum
% between 1e-6 and 1e-5, so zoom in on that decade for a sharper estimate.
noiseValues = logspace(-6,-5,15); % 1e-6 ... 1e-5, ~15 points across one decade
if ~any(abs(log10(noiseValues) - log10(1e-5)) < 1e-9)
    noiseValues = sort([noiseValues, 1e-5]); % make sure the advisor's value is included
end

logML = nan(size(noiseValues));
Ce_ME = nan(size(noiseValues)); % evidence-maximizing noise estimate at the MAP, for sanity checking

for ii = 1:numel(noiseValues)
    config = loadDefaultConfig();
    config.preproc.DSmode = 'rate';
    config.preproc.DSvalue = 3e-4;

    config.inference.includeNoise = true;
    config.inference.dataVariance = noiseValues(ii);

    [~, posterior, modelRanking] = inferImpulseResponse(u_full, q_full, t_full, T_c, ...
        modelOrders=modelOrder, config=config);

    logML(ii) = modelRanking.logML;
    Ce_ME(ii) = posterior.Ce;

    fprintf('noise = %.1e -> logML = %.6g\n', noiseValues(ii), logML(ii));
end

[bestLogML, indBest] = max(logML);
fprintf('\nBester fixierter Noise-Wert: %.1e (logML = %.6g)\n', noiseValues(indBest), bestLogML);

%% Reference runs
% Noise-free baseline (current default in runInference.m)
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;
config.inference.includeNoise = false;
[~, ~, modelRankingNoNoise] = inferImpulseResponse(u_full, q_full, t_full, T_c, ...
    modelOrders=modelOrder, config=config);
logML_noNoise = modelRankingNoNoise.logML;
fprintf('Rauschfrei (includeNoise=false): logML = %.6g\n', logML_noNoise);

% Evidence-maximizing (MML) noise estimate
config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;
config.inference.includeNoise = true;
config.inference.dataVariance = [];
[~, posteriorMML, modelRankingMML] = inferImpulseResponse(u_full, q_full, t_full, T_c, ...
    modelOrders=modelOrder, config=config);
logML_MML = modelRankingMML.logML;
Ce_MML = posteriorMML.Ce;
fprintf('MML-geschaetzter Noise-Wert: Ce = %.3e, logML = %.6g\n', Ce_MML, logML_MML);

%% Plot
figure;
semilogx(noiseValues, logML, 'o-', 'LineWidth', 1.5, 'DisplayName', 'fixierter Noise-Sweep');
hold on
yline(logML_noNoise, '--', 'rauschfrei', 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(1e-5, ':', '1e-5 (Betreuer)', 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
xline(Ce_MML, '-.', 'MML-Schaetzung', 'LabelVerticalAlignment', 'top', 'HandleVisibility', 'off');
plot(noiseValues(indBest), bestLogML, 'p', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'DisplayName', 'bestes Ergebnis');
xlabel('Rauschvarianz C_e')
ylabel('log-Modellevidenz (logML)')
grid on
box on
legend('Location', 'best', 'box', 'off');
title('Noise-Sweep: Modellevidenz vs. fixierte Rauschvarianz')
