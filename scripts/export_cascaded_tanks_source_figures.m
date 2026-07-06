clear;
clc;
close all;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot, 'src')));

sourceDir = fullfile(repoRoot, 'figures', 'source_fig');
outputDir = fullfile(repoRoot, 'figures', 'exported');
ensure_dir(outputDir);

figNames = {'DL_PP2.fig', 'DL_PP.fig', 'DL_LS.fig', 'DL_off.fig', 'DL_mean.fig'};
methodLabels = {'DLM-WLSR2', 'DLM-WLSR1', 'DLM-LS', 'DLM-Off', 'DLM-Mean'};

figure('Color', 'w', 'Position', [100, 100, 2100, 360]);
tiledlayout(1, numel(figNames), 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:numel(figNames)
    figPath = fullfile(sourceDir, figNames{i});
    f = openfig(figPath, 'invisible');

    allAxes = findobj(f, 'Type', 'axes');
    tags = get(allAxes, 'Tag');
    if ischar(tags)
        tags = {tags};
    end
    validAxes = allAxes(~ismember(tags, {'legend', 'colorbar'}));

    if isempty(validAxes)
        close(f);
        warning('No valid axes found in %s. Skipping.', figNames{i});
        continue;
    end

    ax = validAxes(1);
    lines = flipud(findobj(ax, 'Type', 'line'));

    validLines = gobjects(0);
    for j = 1:numel(lines)
        if numel(lines(j).XData) > 10
            validLines(end + 1, 1) = lines(j); %#ok<SAGROW>
        end
    end

    if numel(validLines) < 2
        close(f);
        warning('Fewer than two data lines found in %s. Skipping.', figNames{i});
        continue;
    end

    x = validLines(1).XData;
    yMeasured = validLines(1).YData;
    ySimulated = validLines(2).YData;
    close(f);

    nexttile;
    plot(x, yMeasured, 'b', 'LineWidth', 1.8);
    hold on;
    plot(x, ySimulated, 'r--', 'LineWidth', 1.8);
    xlabel('Time (s)');
    ylabel('Value (V)');
    title(sprintf('(%c) %s', char(96 + i), methodLabels{i}));
    xlim([min(x), max(x)]);
    grid on;
    box on;
    legend({'Measured output', 'Simulated output'}, 'Location', 'southwest');
    set(gca, 'FontName', 'Times New Roman', 'LineWidth', 1, 'FontSize', 11);
end

outputPdf = fullfile(outputDir, 'cascaded_tanks_prediction_comparison.pdf');
outputPng = fullfile(outputDir, 'cascaded_tanks_prediction_comparison.png');

if exist('exportgraphics', 'file') == 2
    exportgraphics(gcf, outputPdf, 'ContentType', 'vector');
else
    print(gcf, outputPdf, '-dpdf', '-bestfit');
end
print(gcf, outputPng, '-dpng', '-r300');

fprintf('Saved Cascaded Tanks source-figure export to %s\n', outputDir);
