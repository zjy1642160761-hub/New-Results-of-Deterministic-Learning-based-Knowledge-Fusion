clc; clear; close all;

%% ====================== 参数设置 ======================
global p1 p2 p3

p_params = [0.2 0.4 2.05; 0.2 0.4 1.95; 0.2 0.4 1.85; ...
            0.2 0.4 1.75; 0.2 0.4 1.65; 0.2 0.4 1.55];

t0 = 0; dt = 0.01; tfinal = 180;
X0 = [0, 0, 0];

%% ====================== 生成多条轨迹 ======================
y = cell(6,1);
y_f3True = cell(6,1);

figure('Name','State Trajectories','Position',[50 50 800 600]);
colors = lines(6); hold on;

for i = 1:6
    p1 = p_params(i,1); p2 = p_params(i,2); p3 = p_params(i,3);
    [~, y{i}] = ode45('rossler', t0:dt:tfinal, X0);

    plot3(y{i}(10000:end,1), y{i}(10000:end,2), y{i}(10000:end,3), ...
          'Color', colors(i,:), 'LineWidth', 1.2);

    y_f3True{i} = p_params(i,2) + y{i}(10000:end,3).*(y{i}(10000:end,1) - p_params(i,3));
end

xlabel('$x_1$','Interpreter','latex'); ylabel('$x_2$','Interpreter','latex'); zlabel('$x_3$','Interpreter','latex');
title('Rossler System State Trajectories'); grid on; view(-120,30);
legend('Traj1','Traj2','Traj3','Traj4','Traj5','Traj6');

%% ====================== RBF 网络布置 ======================
NNDim = 2;
NNRange = [3.5, -2.5; 3, -0.5];
eta = 0.3;
cent = proNNect(NNDim, NNRange, eta);

%% ====================== 学习参数 ======================
propertyData.TS = dt;       propertyData.eta = eta;
propertyData.repeat = 10;   propertyData.gamma = 250;
propertyData.keta = 1;      propertyData.alpha = 0.3;
propertyData.kepsilon = 2;

%% ====================== 确定学习 ======================
W_all = cell(6,1);
WS_all = cell(6,1); % 修复：必须保存所有轨迹的逼近值用于后续空间映射
figure('Name','f3 Approximation','Position',[100 100 1200 800]);

for i = 1:6
    % 修复覆盖问题：将 WS_all 存为 cell 数组
    [W_all{i}, WS_all{i}, ~, ~] = deterministicLearning(y{i}(10000:end,3), ...
        y{i}(10000:end,[1,3]), cent.cent, propertyData);

    subplot(2,3,i);
    plot(y_f3True{i},'b','LineWidth',1.2); hold on;
    plot(WS_all{i},'r--','LineWidth',1.2);
    title(['Trajectory ',num2str(i)]);
    xlabel('Steps'); ylabel('$f_3$');
    legend('True','DL Approx'); grid on;
end

%% ====================== 预计算与数据打包 (加速寻优) ======================
sys.y = y; 
sys.y_f3True = y_f3True; 
sys.W_all = W_all; % 传入逼近的目标函数值
sys.p_params = p_params; 
sys.prop = propertyData;

% 预计算评价eg时的网格与真实值（去除了原本写死的 SSp_grid 激活层计算）
a = -3:0.02:3.5; b = 0:0.01:2;
[A, B] = meshgrid(a, b);
sys.grid_pts = [A(:), B(:)];

sys.grid_true = zeros(6, length(sys.grid_pts));
for k = 1:6
    sys.grid_true(k,:) = p_params(k,2) + B(:)' .* (A(:)' - p_params(k,3));
end

%% ====================== 敏感性分析 (4个超参数) ======================
base_eta_f = 0.3;   
base_keta_f = 1.0; 
base_lambda = 1;    
base_Vb = -2.25;    

list_eta_f = 0.05:0.05:1;
list_keta_f = 0.05:0.05:2;
list_lambda = 1:5:200;
list_Vb = linspace(-1, -4.5, 50);

res_eta_f = zeros(length(list_eta_f), 2);
res_keta_f = zeros(length(list_keta_f), 2);
res_lambda = zeros(length(list_lambda), 2);
res_Vb = zeros(length(list_Vb), 2);

disp('正在进行超参数寻优...');

for i = 1:length(list_eta_f)
    [res_eta_f(i,1), res_eta_f(i,2)] = eval_fusion(list_eta_f(i), base_keta_f, base_lambda, base_Vb, sys);
end

for i = 1:length(list_keta_f)
    [res_keta_f(i,1), res_keta_f(i,2)] = eval_fusion(base_eta_f, list_keta_f(i), base_lambda, base_Vb, sys);
end

for i = 1:length(list_lambda)
    [res_lambda(i,1), res_lambda(i,2)] = eval_fusion(base_eta_f, base_keta_f, list_lambda(i), base_Vb, sys);
end

for i = 1:length(list_Vb)
    [res_Vb(i,1), res_Vb(i,2)] = eval_fusion(base_eta_f, base_keta_f, base_lambda, list_Vb(i), sys);
end
disp('寻优完成！');

%% ====================== 可视化（1x4 大图） ======================
figure('Name','Hyperparameter Tuning Results','Position',[50 150 1800 400]);

% 修改 labels，使用 LaTeX 数学语法
% 注意：在 MATLAB 中使用 latex 解释器，需要用 $...$ 括起来
plot_params = {
    list_eta_f, res_eta_f, '$h$';                % (a) h
    list_keta_f, res_keta_f, '$k_\eta$';          % (b) k*eta
    list_lambda, res_lambda, '$\Gamma scale$';   % (c) Lambda Scale
    list_Vb, res_Vb, '$Threshold$'                 % (d) Activation threshold
};

for idx = 1:4
    subplot(1, 4, idx);
    x_val = plot_params{idx, 1};
    res = plot_params{idx, 2};
    
    yyaxis left
    plot(x_val, res(:,1), '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    ylabel('NAE');
    
    yyaxis right
    plot(x_val, res(:,2), '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    ylabel('NGE');
    
    % 设置 xlabel 并强制使用 latex 解释器
    xlabel(plot_params{idx, 3}, 'Interpreter', 'latex', 'FontSize', 14);
    
    % 标题也可以用 latex 解释器
    title(['(', char(96+idx), ') '], 'Interpreter', 'latex');
    
    grid on;
    legend('NAE', 'NGE', 'Location','best');
end

figure('Name','Hyperparameter Tuning Results','Position',[50 150 1800 400]);

% 修改 labels，使用 LaTeX 数学语法
plot_params = {
    list_eta_f, res_eta_f, '$h$';                % (a) h
    list_keta_f.*0.3, res_keta_f, '$\eta$';          % (b) k*eta
    list_lambda, res_lambda, '$Scale$';   % (c) Lambda Scale
    list_Vb, res_Vb, '$Threshold$'                 % (d) Activation threshold
};

% ================= 设置绝对边距 (0~1 归一化比例) =================
left_margin = 0.05;   % 左侧边缘留白 (防左侧y轴数字被切)
right_margin = 0.05;  % 右侧边缘留白 (防右侧y轴数字被切)
bottom_margin = 0.15; % 底部边缘留白 (留给 xlabel)
top_margin = 0.1;     % 顶部边缘留白 (留给 title)
gap = 0.08;           % 4个子图之间的横向间距

% 自动计算每个子图应有的宽度和高度
width = (1 - left_margin - right_margin - 3 * gap) / 4;
height = 1 - bottom_margin - top_margin;
% =================================================================

for idx = 1:4
    % 替代 subplot，精准分配绘制区域 [左下角X, 左下角Y, 宽度, 高度]
    ax_left = left_margin + (idx - 1) * (width + gap);
    axes('Position', [ax_left, bottom_margin, width, height]);
    
    x_val = plot_params{idx, 1};
    res = plot_params{idx, 2};
    
    yyaxis left
    plot(x_val, res(:,1), '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    ylabel('NAE');
    
    yyaxis right
    plot(x_val, res(:,2), '-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    ylabel('NGE');
    
    % 设置 xlabel 并强制使用 latex 解释器
    xlabel(plot_params{idx, 3}, 'Interpreter', 'latex', 'FontSize', 14);
    
    % 标题也可以用 latex 解释器
    title(['(', char(96+idx), ') '], 'Interpreter', 'latex', 'FontSize', 14);
    
    grid on;
    legend('NAE', 'NGE', 'Location','northeast');
end

%% ====================== 核心评估函数 ======================
function [ea, eg] = eval_fusion(eta_f, keta_f, lambda_scale, Vb_exp, sys)

    NNDim = 2;
    NNRange = [3.5, -2.5; 3, -0.5];
    cent_f = proNNect(NNDim, NNRange, eta_f);
    M_f = size(cent_f.cent,2);
    
    
    cent = proNNect(NNDim, NNRange, 0.3);
    % 1. 动态计算当前参数下的神经元激活状态
    SS_all_n = cell(6,1);
    for k = 1:6
        x_in = sys.y{k}(10000:end, [1,3]);
        diff = reshape(x_in, [], 1, 2) - reshape(cent.cent', 1, [], 2);
        dist_sq = sum(diff.^2, 3);
        SS_temp = exp(-dist_sq' / (0.3^2));
        
        SS_all_n{k} = SS_temp;
    end
    
    SS_all_f = cell(6,1);
    SSf = [];
    for k = 1:6
        x_in = sys.y{k}(10000:end, [1,3]);
        diff = reshape(x_in, [], 1, 2) - reshape(cent_f.cent', 1, [], 2);
        dist_sq = sum(diff.^2, 3);
        SS_temp = exp(-dist_sq' / (keta_f^2 * eta_f^2));
        
        SS_all_f{k} = SS_temp;
        SSf = [SSf, SS_temp];
    end

    
    
    % 2. 协方差与阈值截断 (修复：取代 diag(SSf*SSf') 避免内存爆炸)
    V = sum(SSf.^2, 2); 
    V = sqrt(V) ./ 2000;
    V_b = exp(Vb_exp) * max(V);
    V = diag(V);
    V(V < V_b) = 0;
    
    % 3. 正则化矩阵组装及跨维度映射
    n = M_f;
    base_lam = [0.001; 0.016; 0.001; 0.001; 0.001; 0.001];
    lam_curr = base_lam * lambda_scale;

    H_reg = zeros(n, n);
    B_reg = zeros(n, 1);

    for k = 1:6
        HkTHk = V * SS_all_f{k} * SS_all_f{k}';
        Ak = HkTHk + lam_curr(k) * eye(n);
        
        
        HkTHkB = V * SS_all_f{k} * SS_all_n{k}';

        H_reg = H_reg + HkTHk + lam_curr(k) * eye(n);
        B_reg = B_reg + (HkTHkB + lam_curr(k) * eye(size(HkTHkB))) * sys.W_all{k}';
       

    end

    % 4. QR 分解求解
    [Q,R] = qr(H_reg, 0);
    tol = max(size(R)) * eps(norm(R,'fro'));
    r = sum(abs(diag(R)) > tol);
    if r == n
        w_fused = R \ (Q' * B_reg);
    else
        w_fused = R(1:r, 1:r) \ (Q(:, 1:r)' * B_reg);
        w_fused = [w_fused; zeros(n-r, 1)];
    end

    % 5. 计算 ea (Trajectory Error) - 必须在此步用新参数生成特征矩阵
    errs_train = zeros(6,1);
    for k = 1:6
        y_fusion = w_fused' * SS_all_f{k}; % 直接利用上面算好的 SS_all_f
        true_f3 = sys.y_f3True{k};
        errs_train(k) = mean(abs(true_f3 - y_fusion')./(abs(true_f3)+1));
    end
    ea = mean(errs_train);

    % 6. 计算 eg (Surface Error) - 动态计算网格的 RBF 响应
    diff_grid = reshape(sys.grid_pts, [], 1, 2) - reshape(cent_f.cent', 1, [], 2);
    dist_sq_grid = sum(diff_grid.^2, 3);
    SSp_grid_dynamic = exp(-dist_sq_grid' / (keta_f^2 * eta_f^2));
    
    y_hat_grid = w_fused' * SSp_grid_dynamic;
    err_all_grid = zeros(6, size(y_hat_grid, 2));
    
    for k = 1:6
        true_val = sys.grid_true(k, :);
        err_all_grid(k, :) = abs(y_hat_grid - true_val) ./ (abs(true_val) + 1);
    end
    
  % 取6条轨迹各点误差的最小值
    Z_err_min = min(err_all_grid, [], 1);
    
    % === 加入你的误差阈值截断逻辑 (< 0.3) ===
    Z_valid = Z_err_min(Z_err_min < 0.3);
    
    % 防崩溃保护：如果所有误差都大于 0.3（寻优极差情况），退回全局均值避免产生 NaN
    if isempty(Z_valid)
        eg = mean(Z_err_min);
    else
        eg = mean(Z_valid);
    end
end