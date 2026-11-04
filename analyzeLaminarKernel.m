% Coloured noise kernel for the laminar data
% Figure 5: AC and PSD of the residual with the fitted kernel
%
% Runs analyzeLaminarFIR first (needs its settings and the cached posterior).

analyzeLaminarFIR;

%% Residual of the Gaussian pulse model, as seen by its likelihood
[~, ~, ~, t_max] = generatePrior(nPulse, T_c, config);
sigR = prepareSignals(u_full, q_full, fs, t_max*T_c, config);
sR = sigR.time_domain.coarse;
hR = calculateImpulseResponse(posterior.b, sR.t_h./T_c, T_c);
r = sR.q(sR.valid) - conv(sR.u, hR, 'valid') .* sR.dt;

%% Matern(3/2) x cosine kernel, fitted by maximum likelihood
kern = @(tau, th) exp(2*th(1)) .* (1 + sqrt(3)*tau/exp(th(2))) ...
    .* exp(-sqrt(3)*tau/exp(th(2))) .* cos(exp(th(3))*tau);
tr = (0:numel(r)-1)'*s.dt;
D = abs(tr - tr');
th0 = log([std(r); T_c/3; 2*pi*420]);
thK = fminsearch(@(th) kernelNLL(kern(D, th), r), th0, opts);

%% AC and PSD of residual and kernel
nLag = round(lagMax*T_c/s.dt);
tLag = (0:nLag)'*s.dt;
[acR, lags] = xcorr(r, nLag, 'coeff');
acR = acR(lags >= 0);

[Pr, fW, PrCI] = pwelch(r, win, nOverlap, nWelch, s.fs, 'ConfidenceLevel', 0.95);
okW = fW > 0;
tauK = (-round(10*T_c/s.dt):round(10*T_c/s.dt))'*s.dt;
kernelPSD = @(th) 2*s.dt*real(exp(-1i*2*pi*fW(okW)*tauK') * kern(abs(tauK), th));

%% Plots
red   = [0.85 0.10 0.10];
grey  = [0.75 0.75 0.75];

% Kernel fit (cf. Zimmermann Fig. 6)
figure(5); clf
subplot(2,1,1)
fill([0 lagMax lagMax 0], 1.96/sqrt(numel(r))*[-1 -1 1 1], grey, 'faceAlpha', 0.5, 'edgeColor', 'none'); hold on
p1 = plot(tLag/T_c, acR, 'k', 'lineWidth', 1.2);
p2 = plot(tLag/T_c, kern(tLag, thK)/exp(2*thK(1)), 'color', red, 'lineWidth', 1.5);
yline(0, 'k:');
xlabel('\tau/T_c [-]'); ylabel('AC [-]'); xlim([0 lagMax]); ylim([-0.6 1]); grid on
legend([p1 p2], 'Residual Gaussian pulses', ...
    sprintf('Fitted: l = %.2f T_c, \\omega_c/2\\pi = %.0f Hz', exp(thK(2))/T_c, exp(thK(3))/(2*pi)), ...
    'location', 'northeast')
subplot(2,1,2)
fill([fW(okW); flipud(fW(okW))], [PrCI(okW,1); flipud(PrCI(okW,2))], grey, 'faceAlpha', 0.5, 'edgeColor', 'none'); hold on
loglog(fW(okW), Pr(okW), 'k', 'lineWidth', 1.2);
loglog(fW(okW), kernelPSD(thK), 'color', red, 'lineWidth', 1.5);
set(gca, 'xScale', 'log', 'yScale', 'log');
xlabel('f [Hz]'); ylabel('PSD [Hz^{-1}]'); xlim([fW(2) s.fs/2]); grid on

function nll = kernelNLL(K, r)
    % Negative log-likelihood of r ~ N(0, K).
    [R, p] = chol(K + 1e-12*mean(diag(K))*eye(size(K)));
    if p > 0
        nll = inf;
        return
    end
    z = R' \ r;
    nll = 0.5*(z'*z + 2*sum(log(diag(R))) + numel(r)*log(2*pi));
end
