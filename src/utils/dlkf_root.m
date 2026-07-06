function root = dlkf_root()
%DLKF_ROOT Return the repository root for this MATLAB package.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

end
