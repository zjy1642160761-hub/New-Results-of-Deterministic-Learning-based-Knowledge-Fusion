function [ W_BAR, WS, SS,SSa, Wtimely,etimely] = deterministicLearning( inputData_y, inputData_x,cent, propertyData )
%deterministicLearning 确定学习算法学习部分
%   此处显示详细说明
% 输入：
% inputData_y：输入的ode解出来的解，要学习x几，比如ode解出x1，x2，x3，要学习x几。
% inputData_x:要学习的f对应的状态参数x，比如dx1=-x2-x3,则x2 x3是对应的参数。
%解耦合
% cent：布置好的规则神经网络
% propertyData：需要配置的参数的结构体
% propertyData.TS;
% propertyData.eta;
% propertyData.repeat;
% propertyData.a;
% propertyData.alpha;
% propertyData.sigma;
%
%输出：
%W_BAR：输出的权值矩阵
%V：权值矩阵的变化
%WS:系统动态
%---------------------------------------------------------------------%
%作者：孙庆华
%时间：2017年11月30日
%修改：吴伟明
%时间：2020/6/14


%基本参数
TS = propertyData.TS;
eta = propertyData.eta;   
kk = propertyData.repeat;  
gamma = propertyData.gamma;
keta=propertyData.keta;
alpha=propertyData.alpha;
kepsilon=propertyData.kepsilon
pj =500;   %权值均值区间


[colCent,M]=size(cent);%colCent表示神经网络的维度行数，M表示神经网络个数
[Steps,colInput] = size(inputData_x);
pa = ones(1,M);
if colCent~= colInput
    error('输入数据的维度与神经网络的维度不符！');
end

colCent=size(inputData_y,2);
inputdata_x = single(inputData_x);
inputdata_y= single(inputData_y);
ST_steps = Steps*kk;%%采样数据个数
x_hat = single(zeros(ST_steps,colCent));
S1 = single(zeros(M,1));
W = single(zeros(M,colCent));
V = single(zeros(pj, M,colCent));


Wtimely=zeros(ST_steps,length(W));
etimely=zeros(length(x_hat),1);
%% 计算一次循环的回归向量
for j=1:Steps
    temp = inputdata_x(j,:)'* pa - cent;  
    SS(:,j) = single(exp( -sum(temp.^2) / (keta.^2 * eta.^2)));
    SSa(:,j) = SS(:,j);
    SSa(SSa(:,j)<exp(-kepsilon^2), j) = 0;
end


%% 确定学习主循环
for i=2:ST_steps-1

    
    %% 多次循环时，仅要一次回归向量SS（这里用作ii来作标签）
    ii=mod(i,Steps);
    
    if ii==0
        ii=Steps;
    end
    
    iim=ii-1;
    if iim==0
    iim=Steps;
    end

        %% 更新动态辨识器与网络权值
        x_hat(i,:)=x_hat(i-1,:)+(alpha-1)*(x_hat(i-1,:)-inputdata_y(iim,:))+TS*SS(:,iim)'*W;
        W=W-TS*SS(:,iim)*gamma*(x_hat(i,:)-inputdata_y(ii,:));
        Wtimely(i,:)=W; % A=zeros(2,3);A(:,1)=1
%         etimely(i)=x_hat(i,:)-inputdata_y(ii,:);
        etimely(i)=SS(:,ii)'*W-(0.4+inputdata_x(ii,2).*(inputdata_x(ii,1)-2.4));
%% 平均区间保留权值
    if i >= ST_steps - pj
                V(i + pj - ST_steps + 1,:,:) = W;
    end

end

%% 计算最后平均区间的平均权值
W_BAR = permute(mean(V, 1),[3,2,1]);

%% 计算WS
WS = W_BAR * SS;

end

