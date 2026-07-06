clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

p1 = 0.2;
p2 = 0.4;
p3 = 2.05;
dt = 0.01;
t = 0:dt:2;
X0 = [0, 0, 0];

[~, y] = ode45(@(t, x) rossler_sys(t, x, p1, p2, p3), t, X0);

cent = proNNect(2, [3.5, -2.5; 3, -0.5], 0.5);

propertyData.TS = dt;
propertyData.eta = 0.5;
propertyData.repeat = 1;
propertyData.gamma = 1;
propertyData.keta = 1;
propertyData.alpha = 0.3;
propertyData.kepsilon = 2;

[W, WS] = deterministicLearning(y(:, 3), y(:, [1, 3]), cent.cent, propertyData);

assert(all(isfinite(W(:))), 'Learned weights must be finite.');
assert(all(isfinite(WS(:))), 'Approximated dynamics must be finite.');

fprintf('Rossler smoke test passed. Weight count: %d\n', numel(W));
