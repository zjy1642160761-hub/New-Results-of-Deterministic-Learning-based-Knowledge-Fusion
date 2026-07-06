function WFu = Wfusion7_mod(SSinput, Winput, sigma)

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

dim = size(SSinput{1},1);



H  = zeros(size(SSinput{1},1));
HY = zeros(size(Winput{1}'));

for i = 1:N

    Hi = SSinput{i}';

    H  = H  + Hi' * Hi ...
            + sigma(i) * eye(size(Hi' * Hi));

    HY = HY + Hi' * Hi * Winput{i}' ...
            + sigma(i) * Winput{i}';

end

% ---------------------------------------------------------
% Fusion
% ---------------------------------------------------------

WFu = pinv(H) * HY;

WFu = WFu';

end