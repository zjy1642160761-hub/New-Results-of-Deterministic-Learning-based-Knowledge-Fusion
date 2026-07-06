clear;
clc;
close all;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

fprintf('Running Rossler main-figure reproduction script...\n');
run(fullfile(repoRoot, 'scripts', 'legacy', 'NNpaperfigure.m'));

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
outputDir = fullfile(repoRoot, 'figures', 'exported');
savedFiles = save_all_figures(outputDir, 'rossler_main');

fprintf('Saved %d figures to %s\n', size(savedFiles, 1), outputDir);
