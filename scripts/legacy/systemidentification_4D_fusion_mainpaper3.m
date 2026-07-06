clc
clear
close all

NNDim=4; %神经网络维度
NNRange=[11,2;4,-4;7,0;4,-4]; %神经网络覆盖的区域 （[-8.10]*[-1,15]） 注：为保证完全覆盖，实际布置出来的网络点往往会偏大一圈
eta=1.826;%2699002882462;  %神经元间距
keta=1.49;%6766666832312;
[cent]=proNNect(NNDim,NNRange,eta);
M=length(cent.cent);

%[gamma,keta,eta,ksigma,x_smooth_interval,dx_smooth_interval,u_smooth_interval,du_smooth_interval,repeat]
%[8.77141089485766e-05,1.46000000000000,1.79000000000000,0.000508526409115272,18,10,30,28,320,0.982368139318448,1]
%[0.0137271531659464,1.36766666832312,1.72699002882462,0.00506580074297906,4,2,26,2,30,0.835433759407054,0.965133504405624,16]

% [0.0137947089660763,1.36618908914951,1.72699002882462,0.00488179841871424,4,2,26,2,30,0.836716061284242,0.965133504405624,16]

gamma=8.12e-05;%7271531659464;


ksigma=0.00048;%06580074297906;
sigma=gamma*ksigma;
%sigma=0.0001/8;


figure;
plot(cent.cent(1,:),cent.cent(2,:),'r*')
xlabel('x_1')
ylabel('x_3')
title('RBF networks and State trajectory')


load('dataBenchmark.mat');

pa = ones(1,M);

T=4;
u=uEst;
x=yEst;
steps=length(x);

x_smooth_interval=18;
dx_smooth_interval=10;

x=smooth(x,x_smooth_interval);


dx=(x(2:end)-x(1:end-1))*5;
dx_sm=smooth(dx,dx_smooth_interval);
dx=[0;dx_sm];

u_smooth_interval=30;
du_smooth_interval=25;


u=smooth(u,u_smooth_interval);
du=(u(2:end)-u(1:end-1))*20;
du_sm=smooth(du,du_smooth_interval);
du=[0;du_sm];




stru.pa=pa;
stru.keta=keta;
stru.cent=cent.cent;
stru.eta=eta;

% figure;
% plot(x);
% figure;
% plot(u);

% figure;plot(du)
% 
% figure;
% plot3(x(2:end),x(1:end-1),u(2:end));
% 
% figure;
% plot3(dx,x(2:end),u(2:end));


tic

cou=1;

repeat=315; %orgin

% a=0.835433759407054;
a=0.9966;

tol=2.0;
for i=1:tol
lenx=length(x);

if i<tol
    pani=1;
else
    pani=0;
end

x_sep{i}=x(((i-1)/tol)*lenx+1:(i/tol)*lenx+1*pani);
dx_sep{i}=dx(((i-1)/tol)*lenx+1:(i/tol)*lenx+1*pani);
u_sep{i}=u(((i-1)/tol)*lenx+1:(i/tol)*lenx+1*pani);
du_sep{i}=du(((i-1)/tol)*lenx+1:(i/tol)*lenx+1*pani);


[SS,W,Wnorm]=DL_4D(x_sep{i},dx_sep{i},u_sep{i},du_sep{i},[gamma,ksigma,T,repeat,a],stru);

figure;
plot(Wnorm);

SSinput{i}=SS;
Winput{i}=W';


% SSinput(:,:,i)=SS;
% Winput(:,:,i)=W';
end


if tol==1
else
decay=1;

% W=Wfusion2_mod(SSinput,Winput,decay);
Tsigma=1.122;
W=Wfusion5_mod(SSinput,Winput,Tsigma);
% W=Wfusion2_mod(SSinput,Winput,decay);
W=W';
end
toc


% Wnorm=single([]);
% x_hat=single([]);
% W=single(zeros(M,1));
% SS=single(zeros(M,steps-2));
% WS=single(zeros(1,steps-2));
% steps=length(x);
% 
% 
% for kk=1:repeat
% 
% x_hat(1)=x(1);
% x_hat(2)=x(2);
% 
% e(1)=x_hat(1)-x(1);
% e(2)=x_hat(2)-x(2);
% 
% 
% for i=2:steps-1
%     
%     SS1=S11s([x(i);dx(i);u(i);du(i)],stru);
%     x_hat(i+1)=x(i)+0.65*e(i)+T*W'*SS1;
%     e(i+1)=x_hat(i+1)-x(i+1);
%     W=W-T*(repeat+1-kk)/repeat*gamma*e(i+1)*SS1-(repeat+1-kk)/repeat*sigma*W;    
% 
%     Wnorm(cou)=norm(W);
%     cou=cou+1;
%     
%     if kk==repeat
%     SS(:,i-1)=SS1;
%     WS(i-1)=W'*SS1;
%     end
% end
% 
% 
% end









% figure;
% plot3(cent.cent(1,:),cent.cent(2,:),cent.cent(3,:),'r*');
% hold on
% plot3(x_hat(2:end),(x_hat(2:end)-x_hat(1:end-1))*5,u(2:end));



% figure;
% plot(x_hat);
% hold on
% plot(x);
% 
% difx=x(2:end)-x(1:end-1);
% figure;plot(-e)
% hold on;plot(difx);
% 
% eRMst=sqrt(1/length(e)*sum(e.^2))
% eRMstx2=sqrt(1/length(difx)*sum(difx.^2))
% 
% figure;
% plot(x);


% % % Sindex=setdiff([1:M]',find(Ssum>0.01*std(Ssum)));
% % % gamma2=100;
% % % sigma2=0.001;
% % 
% % 
% % Sindex=[1:M]';
% % gamma2=1;
% % sigma2=0.01;
% % 
% % 
% % gamma2=0.01;
% % sigma2=0.0001;
% % 
% % cou=1;
% % Loss_time=[];
% % for i=1:100
% %     i
% %     LSum=single(zeros(M,1));
% %     Loss=0;
% %     for j=2:steps-1
% %         rannum=(rand(1)-0.5)*2*2*0 ;Delx1=x(j+1)-x(j); Delx2=x(j)-x(j-1);
% %         SS2=S11([x(j)+rannum*Delx2;x(j-1)+rannum*(-Delx1);u(j)]);%(rand(1)-0.5)*2  (Delx1=x(j+1)-x(j);Delx2=x(j)-x(j-1); )
% %         Err=W'*SS2-(x(j+1)-x(j))/T;
% %         LSum=Err*SS2;%LSum+Err*SS2;
% %         
% %         W(Sindex)=W(Sindex)-gamma2*T*LSum(Sindex)-sigma2*W(Sindex);
% %         Wnorm1(cou)=norm(W(Sindex));
% %         cou=cou+1;
% %         Loss=Loss+Err.^2;
% %     end
% % Loss_time(i)=Loss;
% % % W(Sindex)=W(Sindex)-gamma2*T*LSum(Sindex)/steps-sigma2*W(Sindex);
% % % Wnorm1(cou)=norm(W(Sindex));
% %     cou=cou+1;
% % end
% % figure;
% % plot(Wnorm1);
% % 
% % figure;
% % plot(Loss_time)



%% 验证
x_val=yEst;
u_val=uEst;
k2=-0;

% x_val=smooth(x_val,x_smooth_interval);
u_val=smooth(u_val,u_smooth_interval);

du_val=(u_val(2:end)-u_val(1:end-1))*20;
du_val_sm=smooth(du_val,du_smooth_interval);
du_val=[0;du_val_sm];

figure;
plot3(x_val(2:end),x_val(1:end-1),u_val(2:end));

figure;
plot3((x_val(2:end)-x_val(1:end-1))/4*20,x_val(1:end-1),u_val(2:end));

x_hat_val=single([]);
x_hat_val(1)=x_val(1);
x_hat_val(2)=x_val(2);
e_val=single(zeros(1,steps));

for i=2:steps-1
e_val(i)=x_hat_val(i)-x_val(i);    
x_hat_val(i+1)=x_hat_val(i)+k2*e_val(i)+T*W'*S11s([x_hat_val(i);(x_hat_val(i)-x_hat_val(i-1))*5;u_val(i);du_val(i)],stru);  %x2_hat(i+1)=x2_hat(i)+k2*e(i)+T*W2(i)*S1(x1_hat(i),x2(i));
end

difx_val=x_val(2:end)-x_val(1:end-1);
figure;plot(-e_val)
hold on;plot(difx_val);

eRMst_val=sqrt(1/length(e_val)*sum(e_val.^2))
eRMstx2_val=sqrt(1/length(difx_val)*sum(difx_val.^2))

figure;
plot(x_val);
hold on
plot(x_hat_val,'--')

figure;
plot3(cent.cent(1,:),cent.cent(2,:),cent.cent(3,:),'r*');
hold on
plot3(x_hat_val(2:end),(x_hat_val(2:end)-x_hat_val(1:end-1))*5,u_val(2:end),'b');
hold on
plot3(x_val(2:end),(x_val(2:end)-x_val(1:end-1))*5,u_val(2:end),'r');


%% 验证
x_val=yVal;
u_val=uVal;
k2=-0;
% x_val=smooth(x_val,x_smooth_interval);
u_val=smooth(u_val,u_smooth_interval);


du_val=(u_val(2:end)-u_val(1:end-1))*20;
du_val_sm=smooth(du_val,du_smooth_interval);
du_val=[0;du_val_sm];



% figure;
% plot3(x_val(2:end),x_val(1:end-1),u_val(2:end));
% 
% figure;
% plot3((x_val(2:end)-x_val(1:end-1))/4*20,x_val(1:end-1),u_val(2:end));

x_hat_val=single([]);

x_hat_val(1)=x_val(1);
x_hat_val(2)=x_val(2);
e_val=single(zeros(1,steps));

for i=2:steps-1
e_val(i)=x_hat_val(i)-x_val(i);    
x_hat_val(i+1)=x_hat_val(i)+k2*e_val(i)+T*W'*S11s([x_hat_val(i);(x_hat_val(i)-x_hat_val(i-1))*5;u_val(i);du_val(i)],stru);  %x2_hat(i+1)=x2_hat(i)+k2*e(i)+T*W2(i)*S1(x1_hat(i),x2(i));
end

difx_val=x_val(2:end)-x_val(1:end-1);
% figure;plot(-e_val)
% hold on;plot(difx_val);

eRMst_val=sqrt(1/length(e_val)*sum(e_val.^2))
eRMstx2_val=sqrt(1/length(difx_val)*sum(difx_val.^2))

figure;
plot([0:4:(length(x_val)-1)*4],x_val);
hold on
plot([0:4:(length(x_val)-1)*4],x_hat_val,'--')

figure;
plot3(cent.cent(1,:),cent.cent(2,:),cent.cent(3,:),'r*');
hold on
plot3(x_hat_val(2:end),(x_hat_val(2:end)-x_hat_val(1:end-1))*5,u_val(2:end),'b');
hold on
plot3(x_val(2:end),(x_val(2:end)-x_val(1:end-1))*5,u_val(2:end),'r');
% close all