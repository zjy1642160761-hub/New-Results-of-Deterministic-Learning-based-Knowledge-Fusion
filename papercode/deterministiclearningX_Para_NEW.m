function [W_BAR, WS_output, WNorm] = deterministiclearningX_Para_NEW(inputData, NET, propertyData)
% 确定学习主函数 (修正维度匹配版)
% 输入数据: inputData 格式为 [Batch, Dim, Steps]

    [para_num, ~, Steps] = size(inputData);
    model_fun_num = propertyData.model_fun_num;
    relation_fun = propertyData.relation_fun;

    % 提取学习目标和自变量轨迹
    inputdata_fun = inputData(:, model_fun_num, :);      % [Batch, learn_dim, Steps]
    inputdata_relation = inputData(:, relation_fun, :);  % [Batch, relation_dim, Steps]

    TS = propertyData.TS;
    eta = propertyData.eta;
    kk = propertyData.repeat;
    gamma = propertyData.gamma;
    keta = propertyData.keta;
    alpha = propertyData.alpha;
    pj = propertyData.ave;
    sigma = propertyData.sigma;

    [colCent, N_nodes] = size(NET.cent);
    Steps_All = Steps * kk;
    eta_para = keta * eta^2;
    learn_dim = length(model_fun_num);

    if colCent ~= length(relation_fun)
        error('输入数据的维度与神经网络的维度不符！');
    end
    if pj > Steps
        error('权值平均窗口不能超过数据长度！');
    end

    % 时间衰减权重
    if isfield(propertyData, 'timedecay') && propertyData.timedecay
        time_weight_index = (kk - (1:kk) + 1) / kk;
    else
        time_weight_index = ones(1, kk);
    end

    x_hat = single(zeros(para_num, learn_dim));
    WNorm = single(zeros(1, Steps_All));
    
    % MATLAB 中将高维权重存储为 [Batch, learn_dim, 节点数]
    W = single(zeros(para_num, learn_dim, N_nodes));
    W_stor = single(zeros(para_num, learn_dim, N_nodes, pj));
    WS_output = single(zeros(para_num, learn_dim, Steps));

    % 确定学习主循环
    for outi = 1:kk
        timedecay = time_weight_index(outi);
        fprintf('Repeat Learning----> %d <-------Total Num: %d\n', outi, kk);

        for i = 1:Steps-1
            data_temp = inputdata_relation(:, :, i); % [Batch, relation_dim]

            % 1. 计算 RBF 激活矩阵 (SS_matrix: [Batch, N_nodes])
            SS_matrix = zeros(para_num, N_nodes);
            for b = 1:para_num
                dist_sq = sum((data_temp(b, :)' - NET.cent).^2, 1);
                SS_matrix(b, :) = single(exp(-dist_sq / eta_para));
            end

            % 2. 估计器与权值更新
            x_hatN = single(zeros(para_num, learn_dim));
            error_val = single(zeros(para_num, learn_dim));

            for b = 1:para_num
                % 将当前 Batch 的激活向量重塑为 [1, 1, N_nodes] 以匹配 W 的维度
                SS_vec_3d = reshape(SS_matrix(b, :), 1, 1, []);
                
                for d = 1:learn_dim
                    % 取出当前权值向量 [N_nodes, 1] 用于计算输出
                    W_vec = squeeze(W(b, d, :)); 
                    RBF_out = SS_matrix(b, :) * W_vec;

                    % 状态估计更新
                    x_hatN(b, d) = x_hat(b, d) + (alpha - 1) * (x_hat(b, d) - inputdata_fun(b, d, i)) + TS * RBF_out;

                    % 预测误差
                    error_val(b, d) = x_hatN(b, d) - inputdata_fun(b, d, i+1);

                    % --- 核心修正：自适应学习律更新 ---
                    % 计算梯度项并保持在第三维度方向上
                    update_term = gamma * SS_vec_3d * error_val(b, d) + sigma * W(b, d, :);
                    W(b, d, :) = W(b, d, :) - timedecay * TS * update_term;
                end
            end

%             WNorm(1, (outi-1)*Steps + i) = norm(W(:));
            

            % 获取 W 的维度信息
           [rows, ~, ~] = size(W); 
           currentNorm = vecnorm(W, 2, 3);
           WNorm(1:rows, (outi-1)*Steps + i) = currentNorm;
            
            x_hat = x_hatN;

            % 保存最后 pj 步的权值
            if outi == kk && i >= Steps - pj
                idx = mod(i-1, pj) + 1;
                W_stor(:, :, :, idx) = W;
            end
        end

        x_hat = inputdata_fun(:, :, 1);
        currentNorm1 = vecnorm(W, 2, 3);
        WNorm(1:rows, outi * Steps) = currentNorm1
%         WNorm(1, outi * Steps) = norm(W(:));
    end

    % 3. 填补最后一步权值
    if pj == Steps
        W_stor(:, :, :, end) = W;
    end

    % 4. 计算平均权值 W_BAR
    W_BAR = mean(W_stor, 4); 

    % 5. 计算最终输出 WS_output
    if isfield(propertyData, 'Comput_WS') && propertyData.Comput_WS
        fprintf('Calculating WS-----------Waiting...\n');
        for i = 1:Steps
            data_temp = inputdata_relation(:, :, i);
            for b = 1:para_num
                dist_sq = sum((data_temp(b, :)' - NET.cent).^2, 1);
                SS_vec = exp(-dist_sq / eta_para);
                for d = 1:learn_dim
                    W_vec = squeeze(W_BAR(b, d, :));
                    WS_output(b, d, i) = SS_vec * W_vec;
                end
            end
        end
    end
end