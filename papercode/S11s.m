function SS=S11s(x,stru)
SS=single(zeros(length(stru.cent),1));


temp = x* stru.pa - stru.cent;  
% ex_cent=find(sum(abs(temp),1)<=3*sqrt(length(x))*stru.keta*stru.eta);
% SS(ex_cent,1) = single(exp( -sum(temp(:,ex_cent).^2) / (stru.keta * stru.eta.^2)));

SS(:,1)= single(exp( -sum(temp.^2) ./ (stru.keta * stru.eta.^2)));
end