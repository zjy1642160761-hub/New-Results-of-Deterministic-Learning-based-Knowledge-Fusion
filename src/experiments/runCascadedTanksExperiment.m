function result = runCascadedTanksExperiment(methodName, options)
%RUNCASCADEDTANKSEXPERIMENT Run one Cascaded Tanks reproduction experiment.
%
% result = runCascadedTanksExperiment("DLM-WLSR2") trains the deterministic
% learning model, fuses the learned weights, and evaluates free-run
% prediction on both the estimation and validation sets.

if nargin < 1 || isempty(methodName)
    methodName = 'DLM-WLSR2';
end
if nargin < 2 || isempty(options)
    options = struct();
end

config = cascaded_tanks_config(methodName);

if isfield(options, 'repeatOverride') && ~isempty(options.repeatOverride)
    config.repeat = options.repeatOverride;
end

repoRoot = dlkf_root();
dataPath = get_option(options, 'dataPath', ...
    fullfile(repoRoot, 'data', 'cascaded_tanks', 'dataBenchmark.mat'));
showPlots = get_option(options, 'showPlots', false);
maxSamples = get_option(options, 'maxSamples', []);

data = load(dataPath);
if ~isempty(maxSamples)
    data.uEst = data.uEst(1:maxSamples);
    data.yEst = data.yEst(1:maxSamples);
    data.uVal = data.uVal(1:maxSamples);
    data.yVal = data.yVal(1:maxSamples);
end

NNDim = 4;
NNRange = [11, 2; 4, -4; 7, 0; 4, -4];
cent = proNNect(NNDim, NNRange, config.eta);
M = size(cent.cent, 2);

if showPlots
    figure('Name', 'Cascaded Tanks RBF Centers');
    plot(cent.cent(1, :), cent.cent(2, :), 'r*');
    xlabel('x_1');
    ylabel('x_3');
    title('RBF network centers');
    grid on;
end

T = 4;
u = data.uEst;
x = data.yEst;

x = smooth_signal(x, config.xSmoothInterval);
dx = [0; smooth_signal((x(2:end) - x(1:end-1)) * 5, config.dxSmoothInterval)];

u = smooth_signal(u, config.uSmoothInterval);
du = [0; smooth_signal((u(2:end) - u(1:end-1)) * 20, config.duSmoothInterval)];

stru.pa = ones(1, M);
stru.keta = config.keta;
stru.cent = cent.cent;
stru.eta = config.eta;

SSinput = cell(1, config.numSegments);
Winput = cell(1, config.numSegments);
segmentLength = length(x);

tic;
for i = 1:config.numSegments
    if i < config.numSegments
        includeOverlap = 1;
    else
        includeOverlap = 0;
    end

    startIdx = round(((i - 1) / config.numSegments) * segmentLength + 1);
    endIdx = round((i / config.numSegments) * segmentLength + includeOverlap);

    xSeg = x(startIdx:endIdx);
    dxSeg = dx(startIdx:endIdx);
    uSeg = u(startIdx:endIdx);
    duSeg = du(startIdx:endIdx);

    [SS, W] = DL_4D(xSeg, dxSeg, uSeg, duSeg, ...
        [config.gamma, config.ksigma, T, config.repeat, config.alpha], stru);

    SSinput{i} = SS;
    Winput{i} = W';
end

switch config.fusionMethod
    case 'activation_pinv'
        W = Wfusion2_mod(SSinput, Winput, config.decay);
    case 'wlsr1'
        W = Wfusion5_mod(SSinput, Winput, config.fusionSigma);
    case 'regularized_ls'
        W = Wfusion7_mod(SSinput, Winput, config.fusionSigma);
    case 'ls'
        W = Wfusion8_mod(SSinput, Winput, config.fusionSigma);
    case 'mean'
        W = Wfusion9_mod(SSinput, Winput, config.fusionSigma);
    otherwise
        error('Unknown fusion method: %s', config.fusionMethod);
end
W = W';
trainSeconds = toc;

estimation = evaluate_split(data.yEst, data.uEst, W, stru, config, T, showPlots, ...
    'Estimation-set prediction');
validation = evaluate_split(data.yVal, data.uVal, W, stru, config, T, showPlots, ...
    'Validation-set prediction');

result = struct();
result.method = config.label;
result.sourceScript = config.sourceScript;
result.parameters = config;
result.trainSeconds = trainSeconds;
result.weights = W;
result.estimation = estimation;
result.validation = validation;

fprintf('%s | validation RMSE = %.6f | estimation RMSE = %.6f\n', ...
    result.method, result.validation.rmse, result.estimation.rmse);

end

function metrics = evaluate_split(y, u, W, stru, config, T, showPlots, figureName)

xVal = y(:);
uVal = smooth_signal(u(:), config.uSmoothInterval);
duVal = [0; smooth_signal((uVal(2:end) - uVal(1:end-1)) * 20, config.duSmoothInterval)];

steps = length(xVal);
xHat = single(zeros(steps, 1));
xHat(1) = xVal(1);
xHat(2) = xVal(2);
err = single(zeros(steps, 1));
k2 = 0;

for i = 2:steps-1
    err(i) = xHat(i) - xVal(i);
    dxHat = (xHat(i) - xHat(i-1)) * 5;
    basis = S11s([xHat(i); dxHat; uVal(i); duVal(i)], stru);
    xHat(i+1) = xHat(i) + k2 * err(i) + T * W' * basis;
end

deltaX = xVal(2:end) - xVal(1:end-1);

metrics = struct();
metrics.rmse = sqrt(mean(double(err).^2));
metrics.deltaRmse = sqrt(mean(double(deltaX).^2));
metrics.xMeasured = xVal;
metrics.xPredicted = double(xHat);
metrics.error = double(err);

if showPlots
    time = 0:T:(length(xVal)-1) * T;
    figure('Name', figureName);
    plot(time, xVal, 'b', 'LineWidth', 1.4);
    hold on;
    plot(time, xHat, 'r--', 'LineWidth', 1.4);
    xlabel('Time (s)');
    ylabel('Output');
    legend({'Measured output', 'Simulated output'}, 'Location', 'best');
    grid on;
end

end

function config = cascaded_tanks_config(methodName)

key = lower(char(methodName));
key = strrep(key, '_', '-');

config = struct();
switch key
    case {'dlm-off', 'off'}
        config.label = 'DLM-Off';
        config.sourceScript = 'scripts/legacy/systemidentification_4D_fusion_mainpaper.m';
        config.eta = 1.72;
        config.keta = 1.37;
        config.gamma = 0.0135;
        config.ksigma = 0.005;
        config.xSmoothInterval = 4;
        config.dxSmoothInterval = 2;
        config.uSmoothInterval = 26;
        config.duSmoothInterval = 2;
        config.repeat = 30;
        config.alpha = 0.85;
        config.numSegments = 16;
        config.fusionMethod = 'activation_pinv';
        config.decay = 1;
        config.fusionSigma = [];
    case {'dlm-wlsr1', 'wlsr1'}
        config.label = 'DLM-WLSR1';
        config.sourceScript = 'scripts/legacy/systemidentification_4D_fusion_mainpaper3.m';
        config.eta = 1.826;
        config.keta = 1.49;
        config.gamma = 8.12e-05;
        config.ksigma = 0.00048;
        config.xSmoothInterval = 18;
        config.dxSmoothInterval = 10;
        config.uSmoothInterval = 30;
        config.duSmoothInterval = 25;
        config.repeat = 315;
        config.alpha = 0.9966;
        config.numSegments = 2;
        config.fusionMethod = 'wlsr1';
        config.decay = [];
        config.fusionSigma = 1.122;
    case {'dlm-wlsr2', 'wlsr2'}
        config.label = 'DLM-WLSR2';
        config.sourceScript = 'scripts/legacy/systemidentification_4D_fusion_mainpaper6.m';
        config.eta = 1.78;
        config.keta = 1.42;
        config.gamma = 5e-05;
        config.ksigma = 0.002;
        config.xSmoothInterval = 20;
        config.dxSmoothInterval = 10;
        config.uSmoothInterval = 24;
        config.duSmoothInterval = 24;
        config.repeat = 400;
        config.alpha = 0.96;
        config.numSegments = 2;
        config.fusionMethod = 'regularized_ls';
        config.decay = [];
        config.fusionSigma = [1e-03, 1e-03];
    case {'dlm-ls', 'ls'}
        config.label = 'DLM-LS';
        config.sourceScript = 'scripts/legacy/systemidentification_4D_fusion_mainpaper7_LS.m';
        config.eta = 1.8;
        config.keta = 1.45;
        config.gamma = 10e-05;
        config.ksigma = 0.025;
        config.xSmoothInterval = 16;
        config.dxSmoothInterval = 10;
        config.uSmoothInterval = 22;
        config.duSmoothInterval = 10;
        config.repeat = 340;
        config.alpha = 0.8;
        config.numSegments = 2;
        config.fusionMethod = 'ls';
        config.decay = [];
        config.fusionSigma = [0, 0];
    case {'dlm-mean', 'mean'}
        config.label = 'DLM-Mean';
        config.sourceScript = 'scripts/legacy/systemidentification_4D_fusion_mainpaper8_mean.m';
        config.eta = 1.76;
        config.keta = 1.55;
        config.gamma = 7.54030108090289e-05;
        config.ksigma = 0.000670475062206271;
        config.xSmoothInterval = 22;
        config.dxSmoothInterval = 22;
        config.uSmoothInterval = 30;
        config.duSmoothInterval = 24;
        config.repeat = 400;
        config.alpha = 0.95;
        config.numSegments = 2;
        config.fusionMethod = 'mean';
        config.decay = [];
        config.fusionSigma = [0, 0];
    otherwise
        error('Unknown Cascaded Tanks method: %s', methodName);
end

end

function value = get_option(options, fieldName, defaultValue)

if isfield(options, fieldName) && ~isempty(options.(fieldName))
    value = options.(fieldName);
else
    value = defaultValue;
end

end
