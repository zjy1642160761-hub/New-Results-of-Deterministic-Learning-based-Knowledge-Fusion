function savedFiles = save_all_figures(outputDir, prefix)
%SAVE_ALL_FIGURES Save all open MATLAB figures as FIG, PNG, and PDF files.

if nargin < 2 || isempty(prefix)
    prefix = 'figure';
end

ensure_dir(outputDir);
figs = findall(groot, 'Type', 'figure');

if isempty(figs)
    savedFiles = {};
    return;
end

figNums = zeros(numel(figs), 1);
for k = 1:numel(figs)
    if isprop(figs(k), 'Number')
        figNums(k) = figs(k).Number;
    else
        figNums(k) = k;
    end
end
[~, order] = sort(figNums);
figs = figs(order);

savedFiles = cell(numel(figs), 3);
for k = 1:numel(figs)
    baseName = sprintf('%s_%02d', prefix, k);
    figPath = fullfile(outputDir, [baseName, '.fig']);
    pngPath = fullfile(outputDir, [baseName, '.png']);
    pdfPath = fullfile(outputDir, [baseName, '.pdf']);

    savefig(figs(k), figPath);
    print(figs(k), pngPath, '-dpng', '-r300');

    if exist('exportgraphics', 'file') == 2
        exportgraphics(figs(k), pdfPath, 'ContentType', 'vector');
    else
        print(figs(k), pdfPath, '-dpdf', '-bestfit');
    end

    savedFiles(k, :) = {figPath, pngPath, pdfPath};
end

end
