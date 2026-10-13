% Artificial noise sweep: add Gaussian noise epsilon ~ N(0, (k*rms(x))^2)
% to x = u or q and compare the inferred impulse response with the one from the clean data. 
%
% errH     - relative L2 error of h w.r.t. the clean reference
% coverage - fraction of h (where |h_ref| > 5% of max) inside +-2 sigma,
%            ~0.95 if the uncertainty is calibrated
% gain     - sum(n), reference 0.968
%
% model order 2, Ce = 1e-5, mean over 5 seeds


%% Load data
ptd = 'data/Laminar/';
T_c = 4.5e-3; 

rawQ = readmatrix(sprintf('%sheatRelease.csv',ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv',ptd));
t_full = rawU(:,1);
u_full = (rawU(:,2) - mean(rawU(:,2)))/mean(rawU(:,2));
q_full = (rawQ(:,2) - mean(rawQ(:,2)))/mean(rawQ(:,2));

%% Settings
k      = [0.05, 0.1, 0.2]; 
cases  = {'q','u'};
nSeeds = 5;

config = loadDefaultConfig();
config.preproc.DSmode = 'rate';
config.preproc.DSvalue = 3e-4;
config.inference.dataVariance = 1e-5;

[bp,Cb,~,t_max] = generatePrior(2,T_c,config); 
clean = prepareSignals(u_full, q_full, 1/mean(diff(t_full)), t_max*T_c, config);

%% Reference on the clean data
[~,~,hRef] = estimatePosterior(clean, bp, Cb, T_c, config);
sig = abs(hRef.val) > 0.05*max(abs(hRef.val));

%% Sweep
errH     = nan(numel(k), numel(cases), nSeeds);
coverage = nan(size(errH));
gain     = nan(size(errH));

for ik = 1:numel(k)
    for ic = 1:numel(cases)
        x = clean.time_domain.coarse.(cases{ic});
        for is = 1:nSeeds
            rng(is);
            epsilon = k(ik) * rms(x) * randn(size(x)); % epsilon ~ N(0, (k*rms(x))^2)
            signals = clean;
            signals.time_domain.coarse.(cases{ic}) = x + epsilon;

            [b,~,h] = estimatePosterior(signals, bp, Cb, T_c, config);
            errH(ik,ic,is)     = norm(h.val - hRef.val) / norm(hRef.val);
            coverage(ik,ic,is) = mean(abs(h.val(sig) - hRef.val(sig)) <= 2*sqrt(h.var(sig)));
            gain(ik,ic,is)     = sum(b(1:3:end));
        end
    end
end

%% Summary (mean over seeds)
fprintf('\n%6s %5s %8s %9s %7s\n','k','case','errH','coverage','gain');
for ik = 1:numel(k)
    for ic = 1:numel(cases)
        fprintf('%6.2f %5s %8.4f %9.2f %7.3f\n', k(ik), cases{ic}, ...
            mean(errH(ik,ic,:)), mean(coverage(ik,ic,:)), mean(gain(ik,ic,:)));
    end
end
