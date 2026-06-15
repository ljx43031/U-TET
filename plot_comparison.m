% plot_comparison.m
% MATLAB R2015b 兼容版本
% 读取 comparison_data.mat 并将6个子图分别绘制为独立图窗

clear; clc; close all;

set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultTextFontSize', 11);

D = load('comparison_data.mat');
n_algos  = round(D.n_algos);
n_frames = round(D.n_frames);
n_gt     = round(D.n_gt_traj);

% 将下划线替换为短横线，避免 TeX 下标解析
raw_names = D.algo_names;
names = cell(size(raw_names));
for ii = 1:numel(raw_names)
    names{ii} = strrep(raw_names{ii}, '_', '-');
end

colors = [0.00, 0.00, 1.00; 0.00, 0.50, 0.00; 1.00, 0.50, 0.00; 0.00, 0.80, 0.80; 1.00, 0.00, 0.00];

%% ===== 图1: OSPA 随时间变化的距离 =====
figure(1);
set(gcf, 'Name', 'OSPA随时间变化的距离', 'NumberTitle', 'off', 'Position', [50, 500, 800, 450]);
hold on;
h_lines = zeros(1, n_algos);
for i = 1:n_algos
    h_lines(i) = plot(1:n_frames, D.ospa_curves(:, i), 'Color', colors(i,:), 'LineWidth', 1.5, 'DisplayName', names{i});
end
hold off;
xlabel('帧', 'Interpreter', 'none');
ylabel('OSPA 距离 (米)', 'Interpreter', 'none');
title('OSPA 随时间变化的距离', 'Interpreter', 'none');
legend(h_lines, names, 'Location', 'NorthEast', 'FontSize', 9, 'Interpreter', 'none');
grid on; set(gca, 'GridAlpha', 0.3); xlim([1, n_frames]);

%% ===== 图2: 目标数量估计 =====
figure(2);
set(gcf, 'Name', '目标数量估计', 'NumberTitle', 'off', 'Position', [100, 450, 800, 450]);
hold on;
h_gt = plot(1:n_frames, D.card_gt(:), 'k-', 'LineWidth', 2.5, 'DisplayName', '真实航迹数 (GT)');
h_est = zeros(1, n_algos);
for i = 1:n_algos
    h_est(i) = plot(1:n_frames, D.card_est(:, i), 'Color', colors(i,:), 'LineWidth', 1.5, 'DisplayName', names{i});
end
hold off;
xlabel('帧', 'Interpreter', 'none');
ylabel('航迹数量', 'Interpreter', 'none');
title('目标数量估计', 'Interpreter', 'none');
legend([h_gt, h_est], [{'真实航迹数 (GT)'}, names], 'Location', 'NorthEast', 'FontSize', 9, 'Interpreter', 'none');
grid on; set(gca, 'GridAlpha', 0.3); xlim([1, n_frames]);

%% ===== 图3: MOTA / 精确率 / 召回率 =====
figure(3);
set(gcf, 'Name', 'MOTA/精确率/召回率', 'NumberTitle', 'off', 'Position', [150, 400, 750, 480]);
bar_data = [D.mota(:), D.precision_v(:), D.recall_v(:)];
hb = bar(1:n_algos, bar_data, 0.75);
set(hb(1), 'FaceColor', [0.20, 0.40, 0.80]);
set(hb(2), 'FaceColor', [0.20, 0.70, 0.30]);
set(hb(3), 'FaceColor', [0.90, 0.40, 0.20]);
set(gca, 'XTick', 1:n_algos, 'XTickLabel', names, 'XTickLabelRotation', 40, 'TickLabelInterpreter', 'none');
ylabel('评分', 'Interpreter', 'none');
title('MOTA / 精确率 / 召回率', 'Interpreter', 'none');
legend({'MOTA', '精确率', '召回率'}, 'Location', 'Best', 'Interpreter', 'none');
grid on;
set(gca, 'GridAlpha', 0.3, 'Layer', 'top', 'YGrid', 'on', 'XGrid', 'off');
ylo = min(bar_data(:)) - 0.1; yhi = max(bar_data(:)) + 0.12; ylim([ylo, yhi]);
n_groups = 3;
group_width = min(0.8, n_groups / (n_groups + 1.5));
for k = 1:3
    vals = bar_data(:, k);
    for i = 1:n_algos
        xpos = i - group_width/2 + (k-1)*group_width/n_groups + group_width/n_groups/2;
        if vals(i) >= 0
            ypos = vals(i) + 0.015;
        else
            ypos = vals(i) - 0.04;
        end
        text(xpos, ypos, sprintf('%.3f', vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 7.5, 'Interpreter', 'none');
    end
end

%% ===== 图4: OSPA 与延迟对比（双Y轴并排柱状图）=====
figure(4);
set(gcf, 'Name', 'OSPA与延迟对比', 'NumberTitle', 'off', 'Position', [200, 350, 750, 480]);
x_idx = (1:n_algos)';
w = 0.35;
clr_ospa    = [0.27, 0.51, 0.71];
clr_latency = [1.00, 0.50, 0.31];
ax1 = axes;
hold(ax1, 'on');
b1 = bar(ax1, x_idx - w/2, D.avg_ospa(:),   w, 'FaceColor', clr_ospa);
b2 = bar(ax1, x_idx + w/2, D.latency_ms(:), w, 'FaceColor', clr_latency);
hold(ax1, 'off');
set(ax1, 'XTick', x_idx, 'XTickLabel', names, 'XTickLabelRotation', 40, 'TickLabelInterpreter', 'none', 'YColor', clr_ospa, 'XLim', [0.5, n_algos+0.5]);
ylabel(ax1, 'OSPA (米)', 'Interpreter', 'none', 'Color', clr_ospa);
title(ax1, 'OSPA 与延迟对比', 'Interpreter', 'none');
grid(ax1, 'on'); set(ax1, 'GridAlpha', 0.3, 'YGrid', 'on', 'XGrid', 'off');
legend(ax1, b1, {'OSPA'}, 'Location', 'NorthWest', 'Interpreter', 'none');
ax2 = axes('Position', get(ax1, 'Position'), 'YAxisLocation', 'right', 'Color', 'none', 'XTick', [], 'XColor', 'none', 'YColor', clr_latency, 'XLim', [0.5, n_algos+0.5], 'YLim', [0, max(D.latency_ms(:))*1.2]);
ylabel(ax2, '延迟 (毫秒)', 'Interpreter', 'none', 'Color', clr_latency);
hold(ax2, 'on'); b2d = bar(ax2, NaN, NaN, w, 'FaceColor', clr_latency); hold(ax2, 'off');
legend(ax2, b2d, {'延迟'}, 'Location', 'NorthEast', 'Interpreter', 'none');

%% ===== 图5: 三维轨迹 =====
figure(5);
set(gcf, 'Name', '三维轨迹对比', 'NumberTitle', 'off', 'Position', [250, 300, 750, 550]);
hold on;
for i = 1:n_gt
    xi = D.gt_x(:, i); yi = D.gt_y(:, i); zi = D.gt_z(:, i);
    valid = ~isnan(xi);
    if sum(valid) > 1
        if i == 1
            plot3(xi(valid), yi(valid), zi(valid), 'k-', 'LineWidth', 1.5, 'DisplayName', '真实轨迹 (GT)');
        else
            plot3(xi(valid), yi(valid), zi(valid), 'k-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
        end
    end
end
if ~isempty(D.best_pts_x) && numel(D.best_pts_x) > 0
    best_label = ['最优算法: ', strrep(D.best_algo_name, '_', '-')];
    scatter3(D.best_pts_x(:), D.best_pts_y(:), D.best_pts_z(:), 6, 'r', 'filled', 'DisplayName', best_label);
end
hold off;
xlabel('X (米)', 'Interpreter', 'none');
ylabel('Y (米)', 'Interpreter', 'none');
zlabel('Z (米)', 'Interpreter', 'none');
title('三维轨迹（真实轨迹 vs 最优算法）', 'Interpreter', 'none');
legend('show', 'Location', 'Best', 'FontSize', 9, 'Interpreter', 'none');
grid on; view(30, 20);

%% ===== 图6: 综合性能评分 =====
figure(6);
set(gcf, 'Name', '综合性能评分', 'NumberTitle', 'off', 'Position', [300, 250, 700, 480]);
hold on;
for i = 1:n_algos
    barh(i, D.composite(i), 0.80, 'FaceColor', colors(i, :));
end
hold off;
set(gca, 'YTick', 1:n_algos, 'YTickLabel', names, 'TickLabelInterpreter', 'none');
xlabel('综合评分', 'Interpreter', 'none');
title('综合性能评分', 'Interpreter', 'none');
xlim([0, 1.05]); grid on;
set(gca, 'GridAlpha', 0.3, 'XGrid', 'on', 'YGrid', 'off', 'Layer', 'top');
for i = 1:n_algos
    text(D.composite(i)+0.015, i, sprintf('%.3f', D.composite(i)), 'VerticalAlignment', 'middle', 'FontSize', 10, 'Interpreter', 'none');
end

fprintf('\n所有图形已绘制完成 (图1~图6)\n');
