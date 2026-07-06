clear;
clc;
close all;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

fprintf('Running Rossler sensitivity-analysis script. This may take a while...\n');
run(fullfile(repoRoot, 'scripts', 'legacy', 'NNpaperfigure_sen2.m'));

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
outputDir = fullfile(repoRoot, 'figures', 'exported');
savedFiles = save_all_figures(outputDir, 'rossler_sensitivity');

fprintf('Saved %d figures to %s\n', size(savedFiles, 1), outputDir);
