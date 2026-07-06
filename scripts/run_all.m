clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

run(fullfile(repoRoot, 'scripts', 'run_table1_cascaded_tanks.m'));

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));
run(fullfile(repoRoot, 'scripts', 'run_rossler_main_figures.m'));
