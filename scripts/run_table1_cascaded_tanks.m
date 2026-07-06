clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

methods = {'DLM-Mean', 'DLM-Off', 'DLM-LS', 'DLM-WLSR1', 'DLM-WLSR2'};
results = cell(numel(methods), 1);

fprintf('Running Cascaded Tanks Table 1 reproduction...\n');
for k = 1:numel(methods)
    fprintf('\n[%d/%d] %s\n', k, numel(methods), methods{k});
    results{k} = runCascadedTanksExperiment(methods{k}, struct('showPlots', false));
end

method = strings(numel(methods), 1);
source_script = strings(numel(methods), 1);
estimation_rmse = zeros(numel(methods), 1);
validation_rmse = zeros(numel(methods), 1);
estimation_delta_rmse = zeros(numel(methods), 1);
validation_delta_rmse = zeros(numel(methods), 1);
runtime_seconds = zeros(numel(methods), 1);

for k = 1:numel(results)
    method(k) = string(results{k}.method);
    source_script(k) = string(results{k}.sourceScript);
    estimation_rmse(k) = results{k}.estimation.rmse;
    validation_rmse(k) = results{k}.validation.rmse;
    estimation_delta_rmse(k) = results{k}.estimation.deltaRmse;
    validation_delta_rmse(k) = results{k}.validation.deltaRmse;
    runtime_seconds(k) = results{k}.trainSeconds;
end

table1 = table(method, source_script, estimation_rmse, validation_rmse, ...
    estimation_delta_rmse, validation_delta_rmse, runtime_seconds);

disp(table1);

outputDir = fullfile(repoRoot, 'results', 'generated');
ensure_dir(outputDir);
writetable(table1, fullfile(outputDir, 'table1_cascaded_tanks_results.csv'));
save(fullfile(outputDir, 'table1_cascaded_tanks_results.mat'), 'table1', 'results');

fprintf('\nSaved results to %s\n', outputDir);
