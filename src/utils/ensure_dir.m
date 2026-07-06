function ensure_dir(pathName)
%ENSURE_DIR Create a directory if it does not already exist.

if ~exist(pathName, 'dir')
    mkdir(pathName);
end

end
