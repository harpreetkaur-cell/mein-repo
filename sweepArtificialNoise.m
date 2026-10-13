% Artificial noise sweep: add Gaussian noise eps ~ N(0, (level*rms(x))^2)
% to u or q and compare the inferred impulse response with the one from
% the clean data. The noise is added after down-sampling, otherwise the
% anti-aliasing filter would remove most of it.
%
% errH     - relative L2 error of h w.r.t. the clean reference
% coverage - fraction of h (where |h_ref| > 5% of max) inside +-2 sigma,
%            ~0.95 if the uncertainty is calibrated
% gain     - sum(n), reference 0.968
%
% Results (model order 2, Ce = 1e-5, mean over 5 seeds):
%   level  case  errH   coverage  gain
%   5%     q     0.4%   1.00      0.967
%   5%     u     1.1%   0.99      0.966
%   10%    q     0.8%   1.00      0.966
%   10%    u     2.4%   0.71      0.960
%   20%    q     1.5%   0.96      0.964
%   20%    u     5.8%   0.31      0.939
% Noise on q is handled well. Noise on u gives larger errors and too narrow
% uncertainty bands, since the likelihood assumes u is known exactly.

%% Load data
ptd = 'data/Laminar/';
T_c = 4.5e-3; % Flame length / bulk flow velocity

rawQ = readmatrix(sprintf('%sheatRelease.csv',ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv',ptd));
t_full = rawU(:,1);
u_full = (rawU(:,2) - mean(rawU(:,2)))/mean(rawU(:,2));
q_full = (rawQ(:,2) - mean(rawQ(:,2)))/mean(rawQ(:,2));

%% Settings
noiseLevels = [0.05, 0.1, 0.2]; % std(eps)/rms(signal)
cases       = {'q','u'};
nSeeds      = 5;

config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;
config.inference.dataVariance = 1e-5;

[bp,Cb,~,t_max] = generatePrior(2,T_c,config); % Model order selected by the ranking at Ce = 1e-5
clean = prepareSignals(u_full, q_full, 1/mean(diff(t_full)), t_max*T_c, config);

%% Reference on the clean data
[~,~,hRef] = estimatePosterior(clean, bp, Cb, T_c, config);
sig = abs(hRef.val) > 0.05*max(abs(hRef.val));

%% Sweep
errH     = nan(numel(noiseLevels), numel(cases), nSeeds);
coverage = nan(size(errH));
gain     = nan(size(errH));

for il = 1:numel(noiseLevels)
    for ic = 1:numel(cases)
        x = clean.time_domain.coarse.(cases{ic});
        for is = 1:nSeeds
            rng(is);
            signals = clean;
            signals.time_domain.coarse.(cases{ic}) = x + noiseLevels(il)*rms(x)*randn(size(x));

            [b,~,h] = estimatePosterior(signals, bp, Cb, T_c, config);
            errH(il,ic,is)     = norm(h.val - hRef.val) / norm(hRef.val);
            coverage(il,ic,is) = mean(abs(h.val(sig) - hRef.val(sig)) <= 2*sqrt(h.var(sig)));
            gain(il,ic,is)     = sum(b(1:3:end));
        end
    end
end

%% Summary (mean over seeds)
fprintf('\n%6s %5s %8s %9s %7s\n','level','case','errH','coverage','gain');
for il = 1:numel(noiseLevels)
    for ic = 1:numel(cases)
        fprintf('%6.2f %5s %8.4f %9.2f %7.3f\n', noiseLevels(il), cases{ic}, ...
            mean(errH(il,ic,:)), mean(coverage(il,ic,:)), mean(gain(il,ic,:)));
    end
end
