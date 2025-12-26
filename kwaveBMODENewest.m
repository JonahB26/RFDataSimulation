% ========================================================================
% k-Wave B-Mode Ultrasound Simulation — 2D (clean, global-delay version)
% ========================================================================
clearvars; clc; close all; import kwave.*;
tic;
% delete(gcp('nocreate'));          % close any existing thread pool
% parpool("Processes", 8  );          % pick your worker count
for i = 1%:10
    guard_m = 0.002; %2-way 2mm guard
    
    %% ------------------- Linear array (2-D)
    num_elems        = 32;
    
    focus_depth = 20e-3;                     % m
    steer_deg   = 10;                         % 0 for broadside
    
    
    %% ------------------- Build a wide phantom (global coordinates) -------
    num_scan_lines = 128;                    % how many lateral pointings
    
    filename = strcat('FEA_15_Resize.mat');
    % filename = 'P39-W1-S3-T.mat';
    % tumor_type = 1;
    % if tumor_type == 1
    %     inc_mean = 1600; inc_std = 75;
    % else 
    %     inc_mean = 1570; inc_std = 75;
    % end
    
    % [scan_lines_pre, scan_lines_fund_pre, rf_pre, rf_raw_pre,scan_lines_post, ...
    % scan_lines_fund_post, rf_post, rf_raw_post] = GenerateRFFramePair(filename, num_elems, guard_m, steer_deg, focus_depth, ...
    %     num_scan_lines,true, i, false);
    [scan_lines_pre, ~, ~, ~,~, ...
    ~, ~, ~] = GenerateRFFramePair(filename, num_elems, guard_m, steer_deg, focus_depth, ...
        num_scan_lines,false, i, false);
    close all;
end


%% Test Windowing Results
% save("NonWindowResult.mat","scan_lines_pre");
%%
% save('scan_lines_pre.mat','scan_lines_pre');
% save('scan_lines_post.mat', 'scan_lines_post');
% 
% scan_lines_pre = load('scan_lines_pre.mat').scan_lines_pre;
% scan_lines_post = load('scan_lines_post.mat').scan_lines_post;
elapsedTime = toc;
fprintf('Elapsed time: %.2f minutes.\n', elapsedTime/60);

dx = 1.8519e-4;
dy = 1.8519e-4;
elem_width_gdpts = 2; 
scale_factor = 2;
r = load('r.mat').r;

% figure;
% subplot(121);
% % B-mode (fundamental / harmonic)
% horz_axis = (0:length(scan_lines_fund_pre(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
% imagesc(horz_axis, r*1e3, scan_lines_fund_pre.'); axis image; colormap(gray);
% set(gca,'YLim',[5,40]); title('B-mode');
% xlabel('Lateral [mm]'); ylabel('Depth [mm]');
% 
% subplot(122);
% % B-mode (fundamental / harmonic)
% horz_axis = (0:length(scan_lines_fund_post(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
% imagesc(horz_axis, r*1e3, scan_lines_fund_post.'); axis image; colormap(gray);
% set(gca,'YLim',[5,40]); title('B-mode');
% xlabel('Lateral [mm]'); ylabel('Depth [mm]');

% Define the depth range you're interested in (5-40 mm)
depth_min = 5e-3;  % 5 mm in meters
depth_max = 40e-3; % 40 mm in meters

% Find the indices in r that correspond to this depth range
depth_idx = find(r >= depth_min & r <= depth_max);

% Extract the image matrices (transposed to match imagesc display)
% These are the actual matrices being displayed
img_pre = scan_lines_fund_pre(:, depth_idx)';
img_post = scan_lines_fund_post(:, depth_idx)';

save("Frame1_K.mat", "img_pre");
save("Frame2_K.mat", "img_post");

% Create the corresponding axes vectors
horz_axis = (0:size(scan_lines_fund_pre,1)-1) * elem_width_gdpts * dy / scale_factor * 1e3; % mm
depth_axis = r(depth_idx) * 1e3; % mm

% Verify dimensions
fprintf('Pre-image size: %d x %d (depth x lateral)\n', size(img_pre,1), size(img_pre,2));
fprintf('Post-image size: %d x %d (depth x lateral)\n', size(img_post,1), size(img_post,2));
fprintf('Horizontal axis: %.2f to %.2f mm\n', horz_axis(1), horz_axis(end));
fprintf('Depth axis: %.2f to %.2f mm\n', depth_axis(1), depth_axis(end));

% Display to verify
figure;
subplot(121);
figure;
imagesc(horz_axis, depth_axis, img_pre); 
axis image; colormap(gray);
title('B-mode Pre');
xlabel('Lateral [mm]'); ylabel('Depth [mm]');

subplot(122);
imagesc(horz_axis, depth_axis, img_post); 
axis image; colormap(gray);
title('B-mode Post');
xlabel('Lateral [mm]'); ylabel('Depth [mm]');

% Now you can compute strain between img_pre and img_post
% Both matrices have the same size and correspond to the same physical coordinates

%%

% Simple difference between the two images
img_diff = img_post - img_pre;

% Compute spatial gradients of the difference
[grad_lateral, grad_axial] = gradient(img_diff, dx*1e3, dy*1e3);  % Gradients in mm^-1

% Magnitude of the gradient (shows edges/boundaries of movement)
grad_magnitude = sqrt(grad_axial.^2 + grad_lateral.^2);

% Visualization
figure('Position', [100, 100, 1200, 400]);

subplot(131);
imagesc(horz_axis, depth_axis, img_diff);
axis image; colorbar; colormap(gca, 'gray');
title('Image Difference (Post - Pre)');
xlabel('Lateral [mm]'); ylabel('Depth [mm]');

subplot(132);
imagesc(horz_axis, depth_axis, grad_magnitude);
axis image; colorbar; colormap(gca, 'hot');
title('Gradient Magnitude of Difference');
xlabel('Lateral [mm]'); ylabel('Depth [mm]');

subplot(133);
imagesc(horz_axis, depth_axis, abs(img_diff));
axis image; colorbar; colormap(gca, 'hot');
title('Absolute Difference');
xlabel('Lateral [mm]'); ylabel('Depth [mm]');

% Print summary statistics
fprintf('Max absolute difference: %.2f\n', max(abs(img_diff(:))));
fprintf('Mean absolute difference: %.2f\n', mean(abs(img_diff(:))));
fprintf('Max gradient magnitude: %.2f mm^-1\n', max(grad_magnitude(:)));

% Find regions with significant change (potential inclusion movement)
threshold = 3 * std(img_diff(:));  % 3 standard deviations
significant_change = abs(img_diff) > threshold;

fprintf('Percentage of pixels with significant change: %.2f%%\n', ...
        100 * sum(significant_change(:)) / numel(significant_change));
%%

% %% FEA Test
% clc; clearvars; close all;
% % ------------------- Initial vars ---------------
%     guard_m = 0.002; %2-way 2mm guard
% 
%     % ------------------- Linear array (2-D)
%     num_elems        = 32;
% 
%     focus_depth = 20e-3;                     % m
%     steer_deg   = 10;                         % 0 for broadside
% 
% 
%     % ------------------- Build a wide phantom (global coordinates) -------
%     num_scan_lines = 128;                    % how many lateral pointings
%     i = 1;
%     % filename = strcat('FeaData_', num2str(i), '.mat');
%     filename = "FEA_1_Resize.mat";
% % ------------------- Controls -------------------
% DATA_CAST      = 'single';            % 'single' or 'gpuArray-single'
% 
% pml_x_size = 20;                      % depth PML
% pml_y_size = 10;                      % lateral PML
% 
% 
% Nx_eff = 256; Ny_eff = 128;                % target MEDIUM size (excl. PML)
% 
% x_span = 40e-3;                       % depth FOV excluding PML
% c0 = 1540;                            % m/s
% rho0 = 1000;                          % kg/m^3
% 
% [Nx, Ny, dx, dy, kgrid] = GenerateKGrid(Nx_eff, Ny_eff, x_span, pml_x_size, pml_y_size, c0);
% kgrid.Nt = 2500;
% % Medium (background)
% medium.alpha_coeff = 0.75;            % dB/(MHz^y·cm)
% medium.alpha_power = 1.5;
% medium.BonA = 6;                      % enable nonlinearity (THI path uses it)
% 
% % ------------------- Input Signal (velocity drive) -------------------
% source_strength   = 1e6;              % Pa (peak)
% tone_burst_freq   = 1.5e6;            % Hz (fundamental)
% tone_burst_cycles = 4;                % cycles
% % guard_m = 0.002; %2-way 2mm guard
% 
% [burst, burstN, guardN, muteN] = GenerateInputSignal(kgrid,c0, rho0, source_strength, tone_burst_freq, tone_burst_cycles, guard_m);
% 
% % ------------------- Linear array (2-D)
% % num_elems        = 32;
% elem_width_gdpts = 2;                 % grid points per element (centers only for TX/RX)
% kerf_gdpts       = 0;
% 
% % transmit focus
% % focus_depth = 20e-3;                     % m
% % steer_deg   = 0;                         % 0 for broadside
% 
% [x_src_pix, x_src_m, y_centers_pix, y_centers_local_m, tx_apo] = GenerateLinearArray(pml_x_size, dx, dy, Ny, num_elems, elem_width_gdpts, kerf_gdpts);
% 
% % ------------------- Build a wide phantom (global coordinates) -------
% % num_scan_lines = 96;                    % how many lateral pointings
% bg_mean = 1;  bg_std = 0.008;
% % inc_mean = 1585; inc_std = 75;
% % filename = 'P39-W1-S3-T.mat';
% % [sound_speed_map, density_map, Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, bg_mean, bg_std, inc_mean, inc_std, c0, rho0);
% [sound_speed_map, density_map, Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, bg_mean, bg_std, c0, rho0);