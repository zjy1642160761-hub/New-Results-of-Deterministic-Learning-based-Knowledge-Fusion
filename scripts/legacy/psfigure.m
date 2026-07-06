figNames = {'DL_PP2.fig','DL_PP.fig','DL_LS.fig','DL_off.fig','DL_mean.fig'};

% 设置图窗尺寸
figure('Color','w','Position',[100 100 1800 350]);

% 消除外围白边，控制子图间距
tiledlayout(1, 4, 'Padding', 'none', 'TileSpacing', 'compact'); 

for i = 1:4
    % 1. 打开 fig 但不显示，并获取句柄
    f = openfig(figNames{i}, 'invisible');
    
    % 2. 安全获取真实数据坐标轴，过滤掉图例(legend)等伪坐标轴
    all_axes = findobj(f, 'Type', 'axes');
    valid_axes = all_axes(~ismember(get(all_axes, 'Tag'), {'legend', 'colorbar'}));
    if isempty(valid_axes)
        close(f); continue;
    end
    ax = valid_axes(1); % 取出主数据坐标轴

    % 3. 获取曲线对象，并将其翻转为“原本绘制的先后顺序”
    lines = findobj(ax, 'Type', 'line');
    lines = flipud(lines); % 让先画的线在 lines(1)，后画的在 lines(2)

    % 4. 高级防护：寻找包含数据点最多的线条，剔除无用线条
    valid_lines = [];
    for j = 1:length(lines)
        if length(lines(j).XData) > 10 % 假设真实数据点一定大于10个
            valid_lines = [valid_lines; lines(j)];
        end
    end

    % 提取数据
    y_meas = valid_lines(1).YData;
    y_sim  = valid_lines(2).YData;
    x      = valid_lines(1).XData;

    close(f) % 数据提取完毕，关闭原图句柄释放内存

    % 5. 开始在新画布中绘制
    nexttile
    plot(x, y_meas, 'b', 'LineWidth', 2); hold on
    plot(x, y_sim,  'r--', 'LineWidth', 2);

    xlabel('Times (s)','FontSize',12)
    ylabel('Values (Volts)','FontSize',12)

    % 【新增】动态生成 (a), (b), (c), (d) 标题
    % char(97) 对应字母 'a'，所以 char(96+i) 可以完美按顺序生成字母
    title(['(', char(96+i), ')'], 'FontName', 'Times New Roman', 'FontSize', 14);

    % 让 X 轴紧贴数据边缘
    xlim([min(x) max(x)])

    grid on
    box on

    set(gca,'FontName','Times New Roman','LineWidth',1,'FontSize',12)

    % 图例设置
    legend({'measured output','simulated output'}, ...
           'Location','southwest')
end

% 导出矢量图（已将文件名修改为 1x4 匹配你的循环数量）
exportgraphics(gcf, 'merge_1x4.pdf', 'ContentType', 'vector');