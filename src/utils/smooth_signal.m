function y = smooth_signal(x, span)
%SMOOTH_SIGNAL Smooth a vector with MATLAB smooth when available.
%
% The original paper code used smooth(x, span). This wrapper preserves that
% behavior when the Curve Fitting Toolbox is installed and falls back to a
% moving average otherwise.

wasRow = isrow(x);
x = x(:);

if exist('smooth', 'file') == 2
    y = smooth(x, span);
else
    y = movmean(x, span);
end

if wasRow
    y = y.';
end

end
