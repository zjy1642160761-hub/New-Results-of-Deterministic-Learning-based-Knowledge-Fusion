function WFu = Wfusion9_mod(SSinput, Winput, sigma)

% =========================================================
% Generalized Distributed Knowledge Fusion
%
% SSinput : cell array
%           each cell -> state sample matrix
%
% Winput  : cell array
%           each cell -> learned weight matrix
%
% sigma   :
%           scalar  -> same regularization for all agents
%           vector  -> individual regularization
%
% =========================================================

N = size(SSinput,2);

% ---------------------------------------------------------
% sigma processing
% ---------------------------------------------------------
% 
% if length(sigma) == 1
%     sigma = sigma * linspace(1.8,1, N);
% end

% ---------------------------------------------------------
% Dimension
% ---------------------------------------------------------

Wavg = 0;

for i = 1:N
    Wavg = Wavg + Winput{i}';
end

Wavg = Wavg / N;

% ---------------------------------------------------------
% Fusion
% ---------------------------------------------------------

% WFu = pinv(H) * HY;
% 
% WFu = WFu';

WFu = Wavg';

end