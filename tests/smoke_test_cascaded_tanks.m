clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

options = struct();
options.showPlots = false;
options.repeatOverride = 1;
options.maxSamples = 128;

result = runCascadedTanksExperiment('DLM-LS', options);

assert(isfinite(result.validation.rmse), 'Validation RMSE must be finite.');
assert(numel(result.weights) > 0, 'Fused weight vector must not be empty.');

fprintf('Cascaded Tanks smoke test passed. Validation RMSE: %.6f\n', ...
    result.validation.rmse);
