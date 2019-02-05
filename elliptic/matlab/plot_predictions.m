% Make plots because python isn't very good at it.
% (true) outputs are mean-removed

% Test
% dir = '/home/steven/src/python/JCP-experiments/debug/infer_full';
dir = ['/home/steven/src/python/JCP-experiments/results/rev1_experiments/', ...
    'joint_kern_Sum_n256/', ...
    'inv_ss_4_noise_0.008_infer_full'];
% dir = ['/home/steven/src/python/JCP-experiments/results/', ...
%     'resolution_experiment/kern_Sum_n128_infer_full/1'];
% dir = ['/home/steven/src/python/JCP-experiments/results/', ...
%     'adaptive_out_experiment/kern_Sum_RBF_infer_diag_n_64_16_adaptive_1'];

test_list = 1;
forward_backward = 'backward';
include_elliptic_output_mean = true;  % the SYSTEM output
input_plot = 'Plot2DBlocks';  % provided observation; surf, contourf, Plot2DBlocks
output_plot = 'surf';  % target
save_figs = false; 
show_cut = true;

%===============================================================================
% Main compute

% if input('Press 1 to close all: ') == 1; close all; end
close all;
pause(1);

z_in_list = {'in_true', 'in_low', 'in_mean', 'in_hi'};
z_out_list = {'out_true', 'out_low', 'out_mean', 'out_hi'};
forward = strcmp(forward_backward, 'forward');

for t = test_list
    fprintf('Test %i\n', t);
    f0 = sprintf('%s/test%i', dir, t);
    add_mean = ~forward && include_elliptic_output_mean;
    plot_set(f0, z_in_list, add_mean, input_plot, save_figs, show_cut);
    add_mean = forward && include_elliptic_output_mean;
    plot_set(f0, z_out_list, add_mean, output_plot, save_figs, show_cut);
%     end
%     pause;
%     close all;
end

%===============================================================================
% Helper functions

function suc = plot_set(f0, z_list, addmean, plot_func_str, save_figs, show_cut)

nz = numel(z_list);

% Load:
suc = true;
z_data = cell(nz, 1);
for i = 1 : nz
    filename = sprintf('%s_%s.dat', f0, z_list{i});
    if ~exist(filename, 'file')
        warning('Couldn''t find %s; skip.', filename);
        suc = false;
        return;
    end
    z_i = dlmread(filename);
    if size(z_i, 1) ~= size(z_i, 2)  %Python row-major vectorized...
        nsl = sqrt(numel(z_i));
        z_i = reshape(z_i, [nsl, nsl])';
    end
    z_data{i} = z_i;
end

% Apply mean
if addmean
    z_data = add_mean_and_scale(z_data);
end

% limits
z_min = min(z_data{1}(:));
z_max = max(z_data{1}(:));
for i = 2 : nz
    z_min_i = min(z_data{i}(:));
    z_max_i = max(z_data{i}(:));
    if z_min_i < z_min
        z_min = z_min_i;
    end
    if z_max_i > z_max
        z_max = z_max_i;
    end
end

% Plot
plot_func = str2func(plot_func_str);
for i = 1 : nz
    h = figure;
    z_i = z_data{i};
    nsl = size(z_i, 1);
    if strcmp(plot_func_str, 'Plot2DBlocks')
        x_i = linspace(0, 1, nsl)';
        y_i = x_i;
        plot_func(x_i, y_i, z_i, [z_min, z_max]);
    else  % 'contourf, surf
        [x_i, y_i, ~] = UnitSquareGrid(nsl);
        plot_func(x_i, y_i, z_i);
    end
    
    axis equal;
    axis([0, 1, 0, 1]);
    axis off;
    set(gca, 'CLim', [z_min, z_max]);
    if i == 1
        colorbar();
    end
    if save_figs
        pause(1)
        f = sprintf('%s_%s', f0, z_list{i});
        saveas(h, sprintf('%s_matlab.fig', f));
        saveas(h, sprintf('%s_matlab.eps', f), 'epsc');
        saveas(h, sprintf('%s_matlab.png', f));
    end
end

if show_cut
    x_cut = cell(4, 1);
    z_cut = cell(4, 1);
    for i = 1 : 4
        z_i = z_data{i};
        nsl = sqrt(numel(z_i));
        mid = round(nsl / 2 + 0.5);
        x_cut{i} = linspace(0, 1, nsl);
        z_cut{i} = z_i(mid, :);
    end
    h_cut = figure;
    % ASSUME low, hi have same nsl here
    FillBetween(x_cut{2}, z_cut{2}, z_cut{4});
    hold on;
    plot(x_cut{3}, z_cut{3});
    plot(x_cut{1}, z_cut{1});
    if save_figs
        f = sprintf('%s_%s', f0, z_list{1});
        saveas(h_cut, sprintf('%s_cut_matlab.fig', f));
        saveas(h_cut, sprintf('%s_cut_matlab.eps', f), 'epsc');
        saveas(h_cut, sprintf('%s_cut_matlab.png', f));
    end
end

end

function z = add_mean_and_scale(z0)

nz = numel(z0);
z = cell(nz, 1);
for i = 1 : nz
    nsl = size(z0{i}, 2);
    [x_surf, ~, ~] = UnitSquareGrid(nsl);
    m = 1 - x_surf;
    z{i} = z0{i} + m;
end

end
