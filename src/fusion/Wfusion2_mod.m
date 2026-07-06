function WFu=Wfusion2_mod(SSinput,Winput,decay)

SSALL=[];
FALL=[];

for i=1:size(SSinput,2)
SSALL=[SSALL;SSinput{i}'];
FALL=[FALL,Winput{i}*SSinput{i}];
end

FALL=FALL';


SSNORM=[];
for i=1:size(SSALL,2)
SSNORM(i)=norm(SSALL(:,i));
end


% decay=-1.5;
% decay=-0.1;

SSALLindex=1:size(SSALL,2);
SSALLindex(find(SSNORM<=exp(decay)*mean(SSNORM)))=[];
SSALLZERO=SSALL;
SSALLZERO(:,find(SSNORM<=exp(decay)*mean(SSNORM)))=[];


WFu=SSALL'*SSALLZERO*pinv(SSALLZERO'*SSALL*SSALL'*SSALLZERO)*SSALLZERO'*FALL;



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