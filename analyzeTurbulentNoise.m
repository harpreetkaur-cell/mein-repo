% Combustion noise of the unforced turbulent BRS flame (cf. Zimmermann Sec. 5.1)
% Without forcing, the heat release fluctuation q' is pure combustion noise.
%
% The LES was run in 5 restarts (0, 0.055, 0.11, 0.16, 0.21 s). After each
% restart there is a short transient, so the first tSkip of every run is
% discarded and AC and PSD are averaged over the 5 runs.
%
% Figure 1: time series of q' with the discarded parts
% Figure 2: autocorrelation of q' with 95% confidence band (Bartlett)
% Figure 3: PSD of q' with 95% confidence interval and 95% power band
% Figure 4: PSD of u and q (cf. analyzeLaminarSpectra Fig. 1); u is only numerical
%           noise, so no impulse response (Bayesian model or FIR) can be inferred

%% Load data (see concatTurbulentUnforced)
ptd = 'data/BRS_unforced/';
T_c = 4.4e-3;                          % convective time scale of the BRS burner (Yoko)

rawQ = readmatrix(sprintf('%sheatRelease.csv', ptd));
rawU = readmatrix(sprintf('%svelocityRef1.csv', ptd));
t  = rawQ(:,1);
Q  = rawQ(:,2);
fs = 1/mean(diff(t));

%% Settings
tRestart = [0 0.055 0.11 0.16 0.21 t(end)];   % start of each run and end of the data
tSkip    = 2e-3;                       % discarded after each restart (u_ref transient ~1.7 ms)
nWelch   = 2^14;                       % Welch segment length -> df = fs/nWelch ~ 61 Hz
lagMax   = 3;                          % maximum lag of the AC in units of T_c
powerFrac = 0.95;                      % power band (cf. Zimmermann Fig. 7)

%% Valid parts of the 5 runs, normalized q' = (Q - mean)/mean
nRun = numel(tRestart) - 1;
keep = false(size(t));
parts = cell(1, nRun);
for r = 1:nRun
    idx = find(t > tRestart(r) + tSkip & t <= tRestart(r+1));
    keep(idx) = true;
    parts{r} = idx;
end
Qmean = mean(Q(keep));
q = (Q - Qmean)/Qmean;
u = (rawU(:,2) - mean(rawU(keep,2)))/mean(rawU(keep,2));

%% PSD: Welch per run, averaged weighted by the number of segments
win = blackmanharris(nWelch);
nOverlap = round(0.661*nWelch);
[Pqq, f, PqqCI] = runAveragedPSD(q, parts, win, nOverlap, nWelch, fs);
[Puu, ~, PuuCI] = runAveragedPSD(u, parts, win, nOverlap, nWelch, fs);

% 95% power band from the cumulative spectral density
csd = cumsum(Pqq)/sum(Pqq);
fBand = [f(find(csd >= (1-powerFrac)/2, 1)), f(find(csd >= 1-(1-powerFrac)/2, 1))];

%% AC: averaged over the runs, weighted by their length
nLag = round(lagMax*T_c*fs);
tLag = (0:nLag)'/fs;
acq = 0;
nTot = 0;
for r = 1:nRun
    qr = q(parts{r}) - mean(q(parts{r}));
    [ac, lags] = xcorr(qr, nLag, 'coeff');
    acq  = acq + numel(qr)*ac(lags >= 0);
    nTot = nTot + numel(qr);
end
acq = acq/nTot;
% Bartlett's formula for the 95% band of the sample AC
acBand = 1.96*sqrt((1 + 2*[0; 0; cumsum(acq(2:end-1).^2)])/nTot);   % lag k: sum over lags 1..k-1

%% Plots
red  = [0.85 0.10 0.10];
blue = [0.10 0.10 0.85];
grey = [0.75 0.75 0.75];

% Time series
figure(1); clf
plot(t, q, 'color', blue, 'lineWidth', 0.8); hold on
yl = [min(q(keep)) max(q(keep))]*1.2;
for r = 1:nRun
    fill(tRestart(r) + [0 tSkip tSkip 0], yl([1 1 2 2]), grey, 'faceAlpha', 0.6, 'edgeColor', 'none');
end
xlabel('t [s]'); ylabel('q'' [-]'); xlim([0 t(end)]); ylim(yl); grid on
legend('q''', 'discarded (restart)', 'location', 'northeast')

% Autocorrelation (cf. Zimmermann Fig. 4 top)
figure(2); clf
fill([tLag; flipud(tLag)]/T_c, [acBand; -flipud(acBand)], grey, 'faceAlpha', 0.5, 'edgeColor', 'none'); hold on
plot(tLag/T_c, acq, 'color', blue, 'lineWidth', 1.5);
yline(0, 'k:');
xlabel('\tau/T_c [-]'); ylabel('AC [-]'); xlim([0 lagMax]); ylim([-0.6 1]); grid on

% PSD (cf. Zimmermann Fig. 4 bottom and Fig. 7)
figure(3); clf
ok = f > 0;
fShow = [max(fBand(1), f(2)) fBand(2)];   % lower edge kept on the log axis
fill([fShow fliplr(fShow)], [1e-20 1e-20 1 1], blue, 'faceAlpha', 0.07, 'edgeColor', 'none'); hold on
xline(fShow(2), ':', 'color', blue);
fill([f(ok); flipud(f(ok))], [PqqCI(ok,1); flipud(PqqCI(ok,2))], grey, 'faceAlpha', 0.6, 'edgeColor', 'none');
loglog(f(ok), Pqq(ok), 'k', 'lineWidth', 1.2);
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim([f(2) 2e4]); grid on
ylim([min(Pqq(ok & f < 2e4))/3, 3*max(Pqq(ok))])
text(sqrt(prod(fShow)), 2*min(Pqq(ok & f < 2e4)), ...
    sprintf('%d%% power: %.0f - %.0f Hz', round(100*powerFrac), fBand), ...
    'horizontalAlignment', 'center', 'color', blue)

% PSD of u and q (cf. analyzeLaminarSpectra Fig. 1)
figure(4); clf
fill([f(ok); flipud(f(ok))], [PuuCI(ok,1); flipud(PuuCI(ok,2))], blue, 'faceAlpha', 0.3, 'edgeColor', 'none'); hold on
fill([f(ok); flipud(f(ok))], [PqqCI(ok,1); flipud(PqqCI(ok,2))], red, 'faceAlpha', 0.3, 'edgeColor', 'none');
p1 = loglog(f(ok), Puu(ok), 'color', blue, 'lineWidth', 1.2);
p2 = loglog(f(ok), Pqq(ok), 'color', red,  'lineWidth', 1.2);
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim([20 1e4]); grid on
legend([p1 p2], 'Inlet velocity u (unforced)', 'Heat release q (combustion noise)', ...
    'location', 'southwest')

function [P, f, PCI] = runAveragedPSD(x, parts, win, nOverlap, nWelch, fs)
    % Welch PSD of each run, averaged weighted by the number of segments,
    % with the 95% chi-squared confidence interval of the average (as in pwelch).
    P = 0;
    nSeg = 0;
    for r = 1:numel(parts)
        xr = x(parts{r}) - mean(x(parts{r}));
        [Pr, f] = pwelch(xr, win, nOverlap, nWelch, fs);
        K = floor((numel(xr) - nOverlap)/(nWelch - nOverlap));
        P = P + K*Pr;
        nSeg = nSeg + K;
    end
    P = P/nSeg;
    PCI = P .* (2*nSeg ./ chi2inv([0.975 0.025], 2*nSeg));
end
