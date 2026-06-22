function WFu=Wfusion5_mod(SSinput,Winput,sigma)

SSALL=[];
FALL=[];

for i=1:size(SSinput,2)
SSALL=[SSALL;SSinput{i}'];
FALL=[FALL,Winput{i}*SSinput{i}];
end

FALL=FALL';




V=diag(SSinput{1}*SSinput{1}'+SSinput{2}*SSinput{2}');
V=V.^(0.5);
V=diag(V);
% V(V < 0.001) = 0;
V=V./(10*max(V(:)));
H1=V*SSinput{1}*SSinput{1}';
H2=V*SSinput{2}*SSinput{2}';


% sigma=0.01;
H=H1'*H1+sigma*eye(size(H1'*H1))+H2'*H2+0.85*sigma*eye(size(H2'*H2));

% Hh=[H1',H2'];
% H=Hh*Hh'+2*sigma*eye(size(H1'*H1));

%(eye(size(H1'*H1))-Hh*(16*sigma*eye(size(Hh'*Hh))+Hh'*Hh)*Hh')/((16*sigma)*eye(size(H1'*H1)))

HY=H1'*H1*Winput{1}'+sigma*Winput{1}'+H2'*H2*Winput{2}'+0.85*sigma*Winput{2}';
WFu=pinv(H)*(HY);











% SSNORM=[];
% for i=1:size(SSALL,2)
% SSNORM(i)=norm(SSALL(:,i));
% end
% 
% 
% % decay=-1.5;
% % decay=-0.1;
% 
% SSALLindex=1:size(SSALL,2);
% SSALLindex(find(SSNORM<=exp(decay)*mean(SSNORM)))=[];
% SSALLZERO=SSALL;
% SSALLZERO(:,find(SSNORM<=exp(decay)*mean(SSNORM)))=[];
% 
% 
% WFu=SSALL'*SSALLZERO*pinv(SSALLZERO'*SSALL*SSALL'*SSALLZERO)*SSALLZERO'*FALL;

%----------------

% WFu=pinv(SSALL)*FALL;
WFu=WFu';
% WS=W*SSALL';
% 
% 
% 
% WFu=pinv(SSALL'*SSALL)*SSALL'*FALL;
% 
% WFu=WFu';

% V=diag(SSALL'*SSALL);
% V=V.^(0.5);
% V=diag(V);
% 
% WFu=(pinv(SSALL'*SSALL*V*V*SSALL'*SSALL)*SSALL'*SSALL*V*V*SSALL'*FALL);

end