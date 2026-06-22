function loss=sigma_objective_WFu_parallel(sigma,SSinput,Winput,WFu_opt)

WFu=Wfusion6_mod(SSinput,Winput,sigma);

loss=norm(WFu-WFu_opt,'fro')^2;

if isnan(loss) || isinf(loss)
    loss=1e30;
end

end