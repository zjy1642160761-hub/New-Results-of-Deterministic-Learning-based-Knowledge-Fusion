function NET = proNNect_NEW(numDimension, rangeMatrx, width)
% 生成规则的 RBF 神经网络节点网格
% 参数:
%   numDimension: 网络维度 (整数 > 0)
%   rangeMatrx:   [numDimension x 2] 的矩阵，每行为 [max, min]
%   width:        网格间距 (标量 > 0)

    if numDimension < 1
        error('维度必须至少为 1');
    end
    if width <= 0
        error('width 必须大于 0');
    end
    if ~isequal(size(rangeMatrx), [numDimension, 2])
        error('rangeMatrx 必须形如 [numDimension, 2]');
    end

    % 1) 计算每个维度索引范围，加入小偏移 delta 保证边界覆盖
    delta = 0.05;
    % rangeMatrx(:, 1) 是 max, rangeMatrx(:, 2) 是 min
    max_idx = ceil((rangeMatrx(:, 1) + delta) / width);
    min_idx = floor((rangeMatrx(:, 2) - delta) / width);

    % 2) 每维的点数
    numPer = max_idx - min_idx + 1;

    % 3) 生成每维的一维坐标向量
    coords = cell(1, numDimension);
    for i = 1:numDimension
        coords{i} = (min_idx(i) : max_idx(i)) * width;
    end

    % 4) ndgrid 生成多维网格 (等效于 Python 的 meshgrid(..., indexing='ij'))
    grids = cell(1, numDimension);
    [grids{:}] = ndgrid(coords{:});

    % 5) 扁平化拼接各维坐标，得到 [numDimension, 节点总数]
    % 等效于 Python 代码中的 cent.reshape(-1, numDimension).T
    cent = zeros(numDimension, prod(numPer));
    for i = 1:numDimension
        cent(i, :) = grids{i}(:);
    end

    % 包装返回对象
    NET.dim = numDimension;
    NET.width = width;
    NET.rangeMatrx = rangeMatrx;
    NET.numPer = numPer;
    NET.cent_axis = grids;
    NET.cent = cent;       % 核心变量：每一列代表一个神经元的位置
    NET.dimlist = coords;
end