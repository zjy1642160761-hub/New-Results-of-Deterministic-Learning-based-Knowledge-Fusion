clc; clear; close all;

%% ====================== 参数设置 ======================
global p1 p2 p3

% 各轨迹参数
p_params = [0.2 0.4 2.05; 0.2 0.4 1.95; 0.2 0.4 1.85; ...
            0.2 0.4 1.75; 0.2 0.4 1.65; 0.2 0.4 1.55];

% 仿真参数
t0 = 0; dt = 0.01; tfinal = 180;
X0 = [0, 0, 0];

%% ====================== 生成多条轨迹 ======================
y = cell(6,1); y_f3True = cell(6,1);
figure('Name','State Trajectories', 'Position',[100 100 800 600]);
colors = lines(6); hold on;
for i = 1:6
    p1 = p_params(i,1); p2 = p_params(i,2); p3 = p_params(i,3);
    [~, y{i}] = ode45('rossler', [t0:dt:tfinal], X0);
    plot3(y{i}(10000:end,1), y{i}(10000:end,2), y{i}(10000:end,3), ...
          'Color', colors(i,:), 'LineWidth', 1.2);
    y_f3True{i} = p_params(i,2) + y{i}(10000:end,3).*(y{i}(10000:end,1) - p_params(i,3));
end

xlabel('$x_1$', 'Interpreter','latex'); 
ylabel('$x_2$', 'Interpreter','latex'); 
zlabel('$x_3$', 'Interpreter','latex');
title('Rossler System State Trajectories'); grid on;
legend('Traj1','Traj2','Traj3','Traj4','Traj5','Traj6');
view(-120, 30);

%% ====================== RBF 网络布置 ======================
NNDim = 2; NNRange = [3.5, -2.5; 3, -0.5]; eta = 0.3;
cent = proNNect(NNDim, NNRange, eta);

%% ====================== 确定学习参数 ======================
propertyData.TS = dt;
propertyData.eta = eta;
propertyData.repeat = 10;
propertyData.gamma = 250;
propertyData.keta = 1;
propertyData.alpha = 0.3;
propertyData.kepsilon = 2;

%% ====================== 确定学习（每条轨迹） ======================
W_all = cell(6,1); WS_all = cell(6,1); SS_all = cell(6,1); SSa_all = cell(6,1);

figure('Name','f3 Approximation per Trajectory', 'Position',[100 100 1200 800]);
for i = 1:6
    [W_all{i}, WS_all{i}, SS_all{i}, SSa_all{i}] = ...
        deterministicLearning(y{i}(10000:end,3), y{i}(10000:end,[1,3]), cent.cent, propertyData);
    
    subplot(2,3,i);
    plot(y_f3True{i}, 'b', 'LineWidth',1.2); hold on;
    plot(WS_all{i}, 'r--', 'LineWidth',1.2);
    title(['Trajectory ', num2str(i)]);
    xlabel('Steps'); ylabel('$f_3$'); legend('True','DL Approx'); grid on;
end

%% ====================== 激活神经元并集 ======================
union_indices = [];
for i = 1:6
    norms = vecnorm(SSa_all{i}, 2, 2);
    active = find(norms > 0);
    union_indices = union(union_indices, active);
end

% 构建融合矩阵
SSf = []; SSaf = []; Y3_DL = [];
for i = 1:6
    SSa_trim = SSa_all{i}(union_indices, :);
    SSf = [SSf, SS_all{i}];
    SSaf = [SSaf, SSa_trim];
    Y3_DL = [Y3_DL; WS_all{i}'];
end

%% 敏感性分析
NNDim = 2;
NNRange = [3.5, -2.5; 3, -0.5];

eta_f = 0.3;      % 新神经元间距
keta_f = 1;      % 新感受野宽度比例

cent_f = proNNect(NNDim, NNRange, eta);
M_f = size(cent_f.cent, 2);

SS_new = cell(6,1);
SSf = [];
SS_all = cell(6,1);

for k = 1:6
    inputdata_x = y{k}(10000:end,[1,3]);
    Steps = size(inputdata_x,1);

    SS_temp = zeros(M_f, Steps);

    for j = 1:Steps
        temp = inputdata_x(j,:)' * ones(1,M_f) - cent_f.cent;
        SS_temp(:,j) = single(exp(-sum(temp.^2) / (keta_f^2 * eta_f^2)));
    end
    SS_all{k}=SS_temp;
    SSf = [SSf, SS_temp];
end

%% ====================== 权重融合方法对比 ======================
%% ====================== 方法一: 顺序WGCV调整lambda ======================
tic;
V = diag(SSf*SSf');     
V = sqrt(V)./2000;
V_b = exp(-2.25)* max(V);

V = diag(V);
V(V < V_b) = 0;

n = size(SS_all{1}, 1);

lambda=[
0.001
0.016
0.001
0.001
0.001
0.001];

H_reg = zeros(n,n);
B_reg = zeros(n,1);

for k = 1:6
    HkTHk = V * SS_all{k} * SS_all{k}';
    Ak = HkTHk + lambda(k) * eye(n);
    Wk = W_all{k};
    Wk = Wk(:);
    H_reg = H_reg + Ak;
    B_reg = B_reg + Ak * Wk;
end

% QR 稳健求解 H_reg * w_fused = B_reg
[Q, R] = qr(H_reg, 0);

tol = max(size(R)) * eps(norm(R, 'fro'));
r = sum(abs(diag(R)) > tol);

if r == n
    w_fused = R \ (Q' * B_reg);
else
    w_fused = R(1:r,1:r) \ (Q(:,1:r)' * B_reg);
    w_fused = [w_fused; zeros(n-r,1)];
end

elapsed_time = toc;

%% 分布式
H1 = V * SS_all{1} * SS_all{1}';
H2 = V * SS_all{2} * SS_all{2}';
H3 = V * SS_all{3} * SS_all{3}';
H4 = V * SS_all{4} * SS_all{4}';
H5 = V * SS_all{5} * SS_all{5}';
H6 = V * SS_all{6} * SS_all{6}';

n = size(W_all{1}, 2);

it = 1:5000;
g = 0.05;

W1k = double(zeros(n, length(it)+1));
W2k = double(zeros(n, length(it)+1));
W3k = double(zeros(n, length(it)+1));
W4k = double(zeros(n, length(it)+1));
W5k = double(zeros(n, length(it)+1));
W6k = double(zeros(n, length(it)+1));

W1k(:,1) = double(W_all{1}');
W2k(:,1) = double(W_all{2}');
W3k(:,1) = double(W_all{3}');
W4k(:,1) = double(W_all{4}');
W5k(:,1) = double(W_all{5}');
W6k(:,1) = double(W_all{6}');

rr = lambda;

W_iter = cell(6,1);
V_iter = cell(6,1);
M = cell(6,1);
H = {H1,H2,H3,H4,H5,H6};
for k = 1:6
    W_iter{k} = zeros(n, length(it)+1);
    V_iter{k} = zeros(n, length(it)+1);

    W_iter{k}(:,1) = double(W_all{k}');

    A =   double(H{k}' * H{k} + rr(k) * eye(n));
    A = (A + A') / 2;

    M{k} = A \ eye(n);
    M{k} = M{k} / (norm(M{k},2) + eps);
end

M1 =  M{1};M2 =  M{2};M3 =  M{3};M4 =  M{4};M5 =  M{5};M6 =  M{6};

sigma=0.000011;
for ii = it
    W1k(:,ii+1) = W1k(:,ii) + g * M1 * (W2k(:,ii) + W3k(:,ii) + W4k(:,ii) + W5k(:,ii) + W6k(:,ii) - 5*W1k(:,ii))-sigma*W1k(:,ii);
    W2k(:,ii+1) = W2k(:,ii) + g * M2 * (W1k(:,ii) + W3k(:,ii) + W4k(:,ii) + W5k(:,ii) + W6k(:,ii) - 5*W2k(:,ii))-sigma*W2k(:,ii);
    W3k(:,ii+1) = W3k(:,ii) + g * M3 * (W1k(:,ii) + W2k(:,ii) + W4k(:,ii) + W5k(:,ii) + W6k(:,ii) - 5*W3k(:,ii))-sigma*W3k(:,ii);
    W4k(:,ii+1) = W4k(:,ii) + g * M4 * (W1k(:,ii) + W2k(:,ii) + W3k(:,ii) + W5k(:,ii) + W6k(:,ii) - 5*W4k(:,ii))-sigma*W4k(:,ii);
    W5k(:,ii+1) = W5k(:,ii) + g * M5 * (W1k(:,ii) + W2k(:,ii) + W3k(:,ii) + W4k(:,ii) + W6k(:,ii) - 5*W5k(:,ii))-sigma*W5k(:,ii);
    W6k(:,ii+1) = W6k(:,ii) + g * M6 * (W1k(:,ii) + W2k(:,ii) + W3k(:,ii) + W4k(:,ii) + W5k(:,ii) - 5*W6k(:,ii))-sigma*W6k(:,ii);
end

w_iter = double(W4k(:,end));

%% 最终迭代n
V0 = 0;
for i = 1:6
    A = H{i}' * H{i} + rr(i) * eye(n);
    D = W_all{i} - w_fused';
    V0 = V0 + D * A * D';
end

min_eig_values = zeros(1,6);  
max_eig_values = zeros(1,6);  
for i = 1:6
    A = H{i}'*H{i} + rr(i)*eye(n);
    eig_vals = eig(A);                
    min_eig_values(i) = min(eig_vals); 
    max_eig_values(i) = max(eig_vals); 
end

lambda_1 = min(min_eig_values); 
lambda_2 = max(max_eig_values);
epsilon_0 = 0.001;
rho = 0.9935;
n = 2 * log(epsilon_0 * sqrt(lambda_1 / V0)) / log(rho);
n = round(n);

figure(11)
hold on;
plot(0:length(it), vecnorm(W1k), 'LineWidth', 2);
plot(0:length(it), vecnorm(W2k), 'LineWidth', 2);
plot(0:length(it), vecnorm(W3k), 'LineWidth', 2);
plot(0:length(it), vecnorm(W4k), 'LineWidth', 2);
plot(0:length(it), vecnorm(W5k), 'LineWidth', 2);
plot(0:length(it), vecnorm(W6k), 'LineWidth', 2);
final_norm = norm(w_fused);
ylim([0,3.5])
plot(0:length(it), final_norm * ones(1,length(it)+1), 'k--', 'LineWidth', 2.5);
xlabel('Iteration steps $n$', 'Interpreter','latex','FontSize',16);
ylabel('$\|W_{(l),i}(n)\|_2$', 'Interpreter','latex','FontSize',16);
% title('Distributed Weight Iteration', 'Interpreter','latex');
legend('$W_{(1),i}$','$W_{(2),i}$','$W_{(3),i}$','$W_{(4),i}$','$W_{(5),i}$','$W_{(6),i}$','$W_{i}^{*}$', 'Interpreter','latex','Location','best');
grid on; box on;
hold off;

disp(['顺序WGCV融合运行时间: ', num2str(elapsed_time), ' 秒']);
disp('最终lambda = ');
disp(lambda');

%% --------------------------------------------------------------------------------
SSf = single(SSf);
Y3_DL = single(Y3_DL);
% 方法2: 伪逆法
w_pinv = (SSf * SSaf' * pinv(SSaf * SSf' * SSf * SSaf') * SSaf * Y3_DL);

% 方法3: 最小二乘 (LS)
w_ls = (pinv(SSf * SSf') * SSf * Y3_DL);

% 方法4: MEAN (LS)
W_mat = cell2mat(W_all);   
W_mean = mean(W_mat,1);    

elapsed_time = toc;
disp(['程序运行时间: ', num2str(elapsed_time), ' 秒']);

%% ====================== 融合验证（沿6条训练轨迹） ======================
M = size(cent.cent, 2);

methods = {'Iter', 'Pinv', 'LS', 'Mean'};
weights = {w_fused, w_pinv, w_ls, W_mean'}; 

errs_train = zeros(6,4); 

for traj = 1:6
    inputdata_x = y{traj}(10000:end,[1,3]);

    SSp = zeros(M, size(inputdata_x,1));
    for j = 1:size(inputdata_x,1)
        temp = inputdata_x(j,:)' * ones(1,M) - cent.cent;
        SSp(:,j) = exp(-sum(temp.^2) / (propertyData.keta^2 * eta^2));
    end

    for m = 1:4 
        y_fusion = weights{m}' * SSp;
        true_f3 = y_f3True{traj};

        errs_train(traj,m) = mean(abs(true_f3 - y_fusion') ./ ...
                                  (abs(true_f3) + 1));
    end
end

disp('6条训练轨迹上的融合误差：');
disp(array2table(errs_train, ...
    'VariableNames', {'Weighted','Pinv','LS','Mean'}, ... 
    'RowNames', {'Traj1','Traj2','Traj3','Traj4','Traj5','Traj6'}));

meanErrs_path = mean(errs_train,1);

fprintf('训练轨迹平均误差 - 加权法: %.6f | 伪逆法: %.6f | LS法: %.6f | Mean法: %.6f\n', ...
        meanErrs_path(1), meanErrs_path(2), meanErrs_path(3), meanErrs_path(4)); 


%% ====================== 曲面可视化（相对6条训练动力学） ======================
a = -3:0.02:3.5; 
b = 0:0.01:2;

Z_fusion = zeros(4, length(b), length(a)); 
Z_err = zeros(4, length(b), length(a));    

for m = 1:4 
    for i = 1:length(a)
        for j = 1:length(b)

            temp = [a(i); b(j)] * ones(1,M) - cent.cent;
            neu = exp(-sum(temp.^2) / (propertyData.keta^2 * eta^2))';

            Z_fusion(m,j,i) = weights{m}' * neu;
            err_each = zeros(6,1);

            for traj = 1:6
                true_f3_train = p_params(traj,2) + b(j) * (a(i) - p_params(traj,3));
                err_each(traj) = abs(Z_fusion(m,j,i) - true_f3_train) / ...
                                 (abs(true_f3_train) + 1);
            end

            Z_err(m,j,i) = min(err_each);
        end
    end
end

meanErrs = zeros(1,4); 
for m = 1:4 
    Z = squeeze(Z_err(m,:,:));
    meanErrs(m) = mean(Z(Z < 0.3));
end

fprintf('曲面最小相对误差-6训练轨迹共同最小值 - 加权法: %.6f | 伪逆法: %.6f | LS法: %.6f | Mean法: %.6f\n', ...
        meanErrs(1), meanErrs(2), meanErrs(3), meanErrs(4)); 


%% ====================== 误差柱状图：6条训练轨迹 ======================
figure('Name','Training Trajectory Error Comparison', 'Position',[100 100 900 500]);
bar(errs_train, 'grouped');
grid on; box on;
xlabel('Training Trajectories');
ylabel('Mean Relative Error');
title('Fusion Error on Six Training Trajectories');
set(gca, 'XTickLabel', {'Traj1','Traj2','Traj3','Traj4','Traj5','Traj6'});
legend({'Weighted','Pinv','LS','Mean'}, 'Location','best'); 

%% ====================== 平均误差柱状图 ======================
figure('Name','Mean Error Comparison', 'Position',[150 150 600 450]);
bar(meanErrs_path);
grid on; box on;
set(gca, 'XTickLabel', {'Weighted','Pinv','LS','Mean'}); 
ylabel('Mean Relative Error');
title('Average Error over Six Training Trajectories');


%% ====================== 动力学曲面：叠加6条训练轨迹 ======================
figure('Name','Fused Dynamics Surfaces with Training Trajectories', ...
       'Position',[50 50 2000 500]); 

colors = lines(6);

for m = 1:4 
    % 核心修改：重新分配排版比例（无 colorbar 版本）
    % 左右各留 0.04 的边距防裁切，每幅图宽 0.215，图与图之间间隔 0.02
    % 底部留 0.12 给 xyz 标签，顶部留 0.08 给标题 (height = 0.80)
    left = 0.04 + (m-1) * 0.235; 
    ax = axes('Position', [left, 0.12, 0.215, 0.80]); 

    surf(a, b, squeeze(Z_fusion(m,:,:)), 'EdgeColor','none');
    shading interp; 
    colormap(jet); % 去掉了 colorbar
    hold on;

    for traj = 1:6
        plot3(y{traj}(10000:end,1), ...
              y{traj}(10000:end,3), ...
              y_f3True{traj}, ...
              'Color', colors(traj,:), ...
              'LineWidth', 1.5);
    end

    xlabel('$x_1$', 'Interpreter','latex','FontSize',14);
    ylabel('$x_3$', 'Interpreter','latex','FontSize',14);
    zlabel('$f_3$', 'Interpreter','latex','FontSize',14);
    
    % 核心修改：仅保留数值作为标题
    title(sprintf('NAE=%.6f', meanErrs_path(m)), 'Interpreter','latex', 'FontSize',16);

    view(-120,30);
    grid on;
    
    % 彻底清除坐标轴内部附带的紧凑留白
    % 因为我们在 Position 中已经留足了真实边距，这里设为 0 可以让核心曲面撑到最大
    set(ax, 'LooseInset', [0 0 0 0]);
end

%% ====================== 误差曲面：相对6条训练动力学平均误差 ======================
figure('Name','Mean Generalization Error Surfaces over Six Training Dynamics', ...
       'Position',[50 50 2000 500]); 

for m = 1:4 
    % 核心修改：同上，极大化充满画板
    left = 0.04 + (m-1) * 0.235; 
    ax = axes('Position', [left, 0.12, 0.215, 0.80]); 

    surf(a, b, squeeze(Z_err(m,:,:)), 'EdgeColor','none');
    shading interp; colorbar; caxis([0 0.3]);
    hold on;

    for traj = 1:6
        plot3(y{traj}(10000:end,1), ...
              y{traj}(10000:end,3), ...
              0.1 * ones(size(y_f3True{traj})), ...
              'Color', colors(traj,:), ...
              'LineWidth', 1.3);
    end

    xlabel('$x_1$', 'Interpreter','latex','FontSize',14);
    ylabel('$x_3$', 'Interpreter','latex','FontSize',14);
    zlabel('Mean Relative Error', 'FontSize',12);
    
    % 核心修改：仅保留数值作为标题
    title(sprintf('%.6f', meanErrs(m)), 'Interpreter','latex', 'FontSize',16);

    view(-120,30);
    grid on;
    % 清除轴域内部紧凑留白
    set(ax, 'LooseInset', get(ax, 'TightInset'));
end

%% ====================== 函数定义区域 ======================
function [lambda_opt, omega_opt] = wgcv_select_lambda(Phi, y, omega0, lam_min, lam_max)
    y = y(:);
    [U, S, ~] = svd(Phi, 'econ');
    sigma = diag(S);
    sigma2 = sigma.^2;
    m = size(Phi,1);
    r = length(sigma);
    Uy = U' * y;
    y_tail_sq = max(norm(y)^2 - norm(Uy)^2, 0);
    omega = omega0;
    lambda_old = sqrt(lam_min * lam_max);
    opts = optimset('Display','off', ...
                    'TolX',1e-4, ...
                    'TolFun',1e-4, ...
                    'MaxFunEvals',500);
    for iter = 1:5
        gfun = @(lam) wgcv_value(lam, omega, sigma2, Uy, y_tail_sq, m, r);
        lambda_opt = fminbnd(gfun, lam_min, lam_max, opts);
        omega_new = wgcv_update_omega(lambda_old, sigma2, Uy, y_tail_sq, m);
        if isfinite(omega_new) && omega_new > 0
            omega = omega_new;
        end
        lambda_old = lambda_opt;
    end
    omega_opt = omega;
end

function G = wgcv_value(lambda, omega, sigma2, Uy, y_tail_sq, m, r)
    phi = sigma2 ./ (sigma2 + lambda);
    numerator = sum((lambda .* Uy ./ (sigma2 + lambda)).^2) + y_tail_sq;
    denominator = sum((1 - omega) .* phi) + ...
                  sum(lambda ./ (sigma2 + lambda)) + ...
                  (m - r);
    G = m * numerator / (denominator^2 + eps);
end

function omega = wgcv_update_omega(lambda0, sigma2, Uy, y_tail_sq, m)
    h1 = sum((lambda0 .* Uy ./ (sigma2 + lambda0)).^2) + y_tail_sq;
    A = sum(lambda0 .* sigma2 .* (Uy.^2) ./ ((sigma2 + lambda0).^3));
    B = sum(sigma2 ./ ((sigma2 + lambda0).^2));
    C = sum(sigma2 ./ (sigma2 + lambda0));
    omega = (m * A) / (B * h1 + C * A + eps);
end
