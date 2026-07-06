function WFu = Wfusion6_mod(SSinput, Winput, sigma)

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

% ---------------------------------------------------------
% Build adaptive weighting matrix V
% ---------------------------------------------------------

Vtemp = zeros(dim,dim);

for i = 1:N
    Vtemp = Vtemp + SSinput{i} * SSinput{i}';
end

V = diag(Vtemp).^(0.5);
V = V ./ ( 500*max(V(:)));

V_b = exp(-2.25)* max(V);
V(V < V_b) = 0;
V=diag(V);
% V(V < 0.001) = 0;

% if max(V(:)) > 0
% end

% ---------------------------------------------------------
% Build global H and HY
% ---------------------------------------------------------

H  = zeros(size(SSinput{1},1));
HY = zeros(size(Winput{1}'));

for i = 1:N

    Hi = V* SSinput{i} * SSinput{i}';

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