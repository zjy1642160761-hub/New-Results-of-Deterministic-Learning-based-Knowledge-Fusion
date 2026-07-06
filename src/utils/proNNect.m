function [ NET ] = proNNect( numDimension, rangeMatrx, width)
%proNNect 规则的RBF神经网络
%   用于生成规则RBF神经网络，要求网络的维度必须大于1;网格宽度一致，各维神经元个数可以不一致
%输入：
%numDimension：生成的神经网络的维度；
%rangeMatrx：行等于神经网络维度，列表示左右边界;
%并要保证第一列的每行都大于第二列对应的每行
%如果不指定该参数，神经网络归一化为[-1,1]，width默认为0.15
% width：神经网络的间隔
%输出
%cent：输出神经网络结构，包括：神经网络维度、神经元间隔、神经网络范围和神经元布局
%作者：孙庆华
%时间：2017年11月30日
if nargin < 2
    width = 0.15;
    %生成指定神经网络边界的矩阵
    rangeMatrx = ones(1, 2);
    rangeMatrx(:, 2) = -1 * rangeMatrx(:, 2);
elseif nargin < 1
    error('请指定生成神经的维度！');
elseif nargin == 2
    width = 0.15;
    [tempcol, ~] = size(rangeMatrx);
    if  tempcol(1) ~= numDimension
        error('设置生成的神经网络维度与设置神经网络边界的维度不一致')
    end
elseif nargin == 3
    if width <=0 %|| width >= 1
        error('请确认0<width是否满足？')
    end
end

if numDimension < 1
    error('生成神经的维度的维度必须大于1，请指定正确的维度！');
end
if prod(rangeMatrx(:,1) - rangeMatrx(:,2)) <= 0
   error('rangeMatrx的范围有误！'); 
end

NET.width = width;
NET.rangeMatrx = rangeMatrx;
NET.dim = numDimension;


rangeNum = zeros(numDimension, 2);

delta=0.05;
rangeNum(:,1) = ceil((rangeMatrx(:, 1) + delta) / width);
rangeNum(:,2) = floor((rangeMatrx(:, 2) - delta) / width);

numPer = rangeNum(:, 1) - rangeNum(:, 2) + 1;%每一维的神经元个数
numPer = [numPer; 1];
cent = single(zeros(prod(numPer), numDimension));%最终生成的神经网络

for i = 1 : numDimension                          %生成网络的维度
    for k = 0 : numPer(i) : prod(numPer(1:i)) - 1       %需要重复的次数
        for j = rangeNum(i, 2) : rangeNum(i, 1)   %设置值的位置
            cent((k + j -(rangeNum(i, 2))) * prod(numPer(i + 1 : numDimension)) + 1 : (k + j -(rangeNum(i, 2)) + 1) * prod(numPer(i +1 : numDimension)), i)...
                = ones(prod(numPer(i + 1:numDimension)), 1, 'single') * j * width;
        end
    end
end

% % for i = 1 : numDimension                          %生成网络的维度
% %     for k = 0 : numPer(i) : numPer(i)^i - 1       %需要重复的次数
% %         for j = rangeNum(i, 2) : rangeNum(i, 1)   %设置值的位置
% %             cent((k + j -(rangeNum(i, 2))) * prod(numPer(i:numDimension)) + 1 : (k + j -(rangeNum(i, 2)) + 1) * prod(numPer(i:numDimension)), i)...
% %                 = ones(prod(numPer(i:numDimension)), 1, 'single') * j * width;
% %         end
% %     end
% % end

NET.cent = cent';

end

