%% 确定学习与动力学融合 (Rossler系统) - Fusion_V2 版本
clear; clc; close all;

%% 1. 初始化与参数设置
p1 = 0.2;
p2 = 0.4;
p3_index = [2.05;1.95;1.85;1.75;1.65;1.55]; % 大规模数据（共101个点）

num_p3 = length(p3_index);
train_Data = cell(num_p3, 1);
train_true_Data = cell(num_p3, 1);

dt = 0.01;
tfinal = 18000 * dt;
t = 0 : dt : (tfinal - dt);
X0 = [0; 0; 0];

%% 2. 轨迹生成
fprintf('正在生成训练轨迹 (Rossler 系统)...\n');
for i = 1:num_p3
    p3 = p3_index(i);
    [~, y] = ode45(@(t, x) rossler_sys(t, x, p1, p2, p3), t, X0);
    
    y_steady = y(10001:end, :); % 截取稳态数据
    train_Data{i} = y_steady;
    train_true_Data{i} = p2 + y_steady(:,3) .* (y_steady(:,1) - p3);
end

%% 3. 神经网络布局与可视化
figure('Color', 'w', 'Name', '神经网络布局');
hold on;
for i = 1:num_p3
    plot(train_Data{i}(:, 1), train_Data{i}(:, 3), 'Color', [0.7 0.7 0.7]);
end

NNDim = 2; NNRange = [3.5, -2.5; 3, -0.5]; eta = 0.3;
cent = proNNect_NEW(NNDim, NNRange, eta);

scatter(cent.cent(1, :), cent.cent(2, :), 10, 'r', 'filled');
xlabel('x1 (Dim 1)'); ylabel('x3 (Dim 3)');
title('Trajectories and Neural Network Layout');
grid on; hold off;

%% 4. 确定学习 (Deterministic Learning)
PropertyData = struct();
PropertyData.TS = dt;
PropertyData.eta = eta;
PropertyData.repeat = 10;
PropertyData.gamma = 250;
PropertyData.keta = 1;
PropertyData.alpha = 0.3;
PropertyData.ave = 500;
PropertyData.timedecay = 0;
PropertyData.sigma = 0;
PropertyData.model_fun_num = 3;     % MATLAB 索引 3
PropertyData.relation_fun = [1, 3]; % 第一、三维
PropertyData.GPU = 0;
PropertyData.Comput_WS = 1;

[N_steady, ~] = size(train_Data{1});
train_data_mat = zeros(num_p3, 3, N_steady);
for i = 1:num_p3
    train_data_mat(i, :, :) = train_Data{i}';
end

fprintf('\n开始并行确定学习...\n');
tic;
[W_output, WS_output, WNorm] = deterministiclearningX_Para_NEW(train_data_mat, cent, PropertyData);
toc;
% 
% load('learningF.mat') % 加载 WS_output
% load('learningW.mat') % 加载 W_output

WS_new = squeeze(WS_output);          % 拟合值（f3近似值）
WSoutput_new = squeeze(W_output);     % 权重矩阵

%% 8. 选择6个不重复的轨迹索引 (修改为 6 个)
sample_num = min(6, num_p3); 
selected_idx = [1,2,3,4,5,6];%randperm(num_p3, sample_num); 

%% 9. 预计算3D曲面网格
fprintf('\n正在计算 3D 动力学曲面...\n');
% [3.5, -2.5; 3, -0.5]
a_axis = -3.5 : 0.05 : 4;  % x1轴范围
b_axis = -0.5 : 0.05 : 2.5;  % x3轴范围
[A, B] = meshgrid(a_axis, b_axis);
z13 = zeros(size(A));    % 存储3D曲面值

%% 10. 创建12个子图的大图（2行6列排列，消除左右空白）
% 适当调整窗口宽度，使其适合6列
figure('Name', '6组轨迹拟合+3D曲面', 'Position', [100, 100, 2200, 800], 'Color', 'w');

% --- 自定义布局参数 (归一化坐标 0~1) ---
cols = 6; 
rows = 2;
L_margin = 0.03;  % 左侧边缘留白 (仅留给第一个图的Y轴数字)
R_margin = 0.01;  % 右侧边缘极限压缩
B_margin = 0.06;  % 底部留白 (X轴标签)
T_margin = 0.06;  % 顶部留白 (给 sgtitle 留出空间)
gap_x = 0.018;    % 子图之间的水平间距
gap_y = 0.12;     % 第一行和第二行之间的垂直间距

% 计算每个子图的宽和高
W = (1 - L_margin - R_margin - (cols-1)*gap_x) / cols;
H = (1 - B_margin - T_margin - (rows-1)*gap_y) / rows;

for k = 1:sample_num
    idx = selected_idx(k);
    p3_val = p3_index(idx); 
    
    % 当前列的起始 X 坐标
    x_pos = L_margin + (k-1)*(W + gap_x);
    
    % -------------------- 第一行：拟合曲线 --------------------
    y_pos_row1 = B_margin + H + gap_y;
    axes('Position', [x_pos, y_pos_row1, W, H]); % 手动指定坐标替换 subplot
    
    plot(train_true_Data{idx}, 'b-', 'LineWidth', 2);
    hold on;
    plot(WS_new(idx,:), 'r--', 'LineWidth', 2);
    
    xlabel('steps', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('$f_3$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
   % title(sprintf('Traj %d ($p_3=%.2f$)', k, p3_val), ...
%           'Interpreter', 'latex', 'FontSize', 14, 'FontName', 'Times New Roman');

    % 计算当前轨迹的逼近误差（RMSE）
    err_rmse = mean(abs(train_true_Data{idx}(:) - WS_new(idx,:).')./(abs( WS_new(idx,:).')+ones(size( WS_new(idx,:).'))));
%   mean(abs(y1_f3True-WS1')./(abs(WS1')+ones(size(WS1'))))
    title(sprintf('NAE=%.4f', err_rmse), ...
    'FontSize', 20, ...
    'FontName', 'Microsoft YaHei', ...
    'Interpreter', 'none');

    legend('$f_3(true)$','$f_3(approx)$', ...
           'Interpreter', 'latex', 'FontSize', 10, 'Location', 'northeast', 'FontName', 'Times New Roman');
    grid on;
    set(gca, 'FontSize', 10, 'FontName', 'Times New Roman');
    
    % -------------------- 第二行：3D曲面+轨迹 --------------------
    y_pos_row2 = B_margin;
    axes('Position', [x_pos, y_pos_row2, W, H]); % 手动指定坐标替换 subplot
    
    for i = 1:numel(A)
        dist_sq = sum(([A(i); B(i)] - cent.cent).^2, 1);
        neu3 = exp(-dist_sq / (eta^2));
        z13(i) = WSoutput_new(idx,:) * neu3';
    end
    
    h_surf = surf(A, B, z13, 'EdgeColor', 'none', 'FaceAlpha', 0.9, 'FaceColor', 'interp');
    set(h_surf, 'DisplayName', sprintf('$\\bar{W}_{(%d),3}^T S(\\cdot)$', idx));
    zlim([-4,4]);          
    caxis([-4,4]);         
    try
        colormap(turbo);   
    catch
        colormap(parula);  
    end
    
    hold on;
    x_traj = train_Data{idx}(:,1);
    y_traj = train_Data{idx}(:,3);
    z_traj = train_true_Data{idx};
    h_traj = plot3(x_traj, y_traj, z_traj, 'r', 'LineWidth', 2);
    set(h_traj, 'HandleVisibility', 'off'); 
    
    xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('$x_3$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
    zlabel('$f_3$', 'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(h_traj, 'DisplayName', sprintf('$f_3(x,p_3=%.3f)$', p3_val)); % 轨迹图例名称
%     title(sprintf('3D ($p_3=%.2f$)', p3_val), ...
%           'Interpreter', 'latex', 'FontSize', 14, 'FontName', 'Times New Roman');
    lgd = legend([h_surf, h_traj], 'Interpreter', 'latex', 'FontSize', 10, ...
                 'Location', 'northeast', 'FontName', 'Times New Roman');
    view(220, 10);  
    grid on;
    set(gca, 'FontSize', 10, 'FontName', 'Times New Roman');
    hold off;
end

% 总标题
% sgtitle('6 Trajectories: Fitting Curves & 3D Dynamics Surfaces', ...
%         'FontSize', 18, 'FontName', 'Times New Roman', 'FontWeight', 'bold');