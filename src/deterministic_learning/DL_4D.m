function [SS,W,Wnorm]=DL_4D(x,dx,u,du,para,stru)



gamma=para(1);
ksigma=para(2);
T=para(3);
repeat=para(4);
a=para(5);


sigma=gamma*ksigma;
steps=length(x);

M=length(stru.cent);

Wnorm=single([]);
x_hat=single([]);
W=single(zeros(M,1));
SS=single(zeros(M,steps-2));
WS=single(zeros(1,steps-2));

cou=1;

for kk=1:repeat

x_hat(1)=x(1);
x_hat(2)=x(2);

e(1)=x_hat(1)-x(1);
e(2)=x_hat(2)-x(2);


for i=2:steps-1
    
    if kk==1
    SS1=S11s([x(i);dx(i);u(i);du(i)],stru);
    SS(:,i-1)=SS1;
    else
    SS1=SS(:,i-1);
    end
    x_hat(i+1)=x(i)+a*e(i)+T*W'*SS1;
    e(i+1)=x_hat(i+1)-x(i+1);
    W=W-T*(repeat+1-kk)/repeat*gamma*e(i+1)*SS1-(repeat+1-kk)/repeat*sigma*W;    

    Wnorm(cou)=norm(W);
    cou=cou+1;
    
end


end

