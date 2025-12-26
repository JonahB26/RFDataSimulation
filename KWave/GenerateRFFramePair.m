function [scan_lines_pre, scan_lines_fund_pre, rf_pre, rf_raw_pre,scan_lines_post, ...
    scan_lines_fund_post, rf_post, rf_raw_post] = GenerateRFFramePair(filename, num_elems, guard_m, steer_deg, focus_depth, num_scan_lines, visualize, i, generate_frame_pair)
%% ------------------- Controls -------------------
DATA_CAST      = 'single';            % 'single' or 'gpuArray-single'

pml_x_size = 20;                      % depth PML
pml_y_size = 10;                      % lateral PML


Nx_eff = 256; Ny_eff = 128;                % target MEDIUM size (excl. PML)

x_span = 40e-3;                       % depth FOV excluding PML
c0 = 1540;                            % m/s
rho0 = 1000;                          % kg/m^3

[Nx, Ny, dx, dy, kgrid] = GenerateKGrid(Nx_eff, Ny_eff, x_span, pml_x_size, pml_y_size, c0);
kgrid.Nt = 2500;

%% ------------------- Input Signal (velocity drive) -------------------
source_strength   = 1e6;              % Pa (peak)
tone_burst_freq   = 1.5e6;            % Hz (fundamental)
tone_burst_cycles = 4;                % cycles
% guard_m = 0.002; %2-way 2mm guard

[burst, burstN, guardN, muteN] = GenerateInputSignal(kgrid,c0, rho0, source_strength, tone_burst_freq, tone_burst_cycles, guard_m);

%% ------------------- Linear array (2-D)
% num_elems        = 32;
elem_width_gdpts = 2;                 % grid points per element (centers only for TX/RX)
kerf_gdpts       = 0;

% transmit focus
% focus_depth = 20e-3;                     % m
% steer_deg   = 0;                         % 0 for broadside

[x_src_pix, x_src_m, y_centers_pix, y_centers_local_m, tx_apo] = GenerateLinearArray(pml_x_size, dx, dy, Ny, num_elems, elem_width_gdpts, kerf_gdpts);

%% ------------------- Build a wide phantom (global coordinates) -------
% num_scan_lines = 96;                    % how many lateral pointings
bg_mean = 1;  bg_std = 0.008;
% inc_mean = 1585; inc_std = 75;
% filename = 'P39-W1-S3-T.mat';
% [sound_speed_map, density_map, Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, bg_mean, bg_std, inc_mean, inc_std, c0, rho0);
[sound_speed_map_pre, density_map_pre, ...
          sound_speed_map_post, density_map_post, ...
          Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, ...
                                           num_scan_lines, elem_width_gdpts, ...
                                           bg_mean, bg_std, c0, rho0);
%% ------------------- Storage + solver args ---------------------------
[scan_lines_pre, rf_pre, rf_raw_pre] = GenerateRFLines(Nx, Ny, dy, num_scan_lines, sound_speed_map_pre, density_map_pre, y_centers_local_m, focus_depth, x_src_m, ...
    c0, kgrid, DATA_CAST, num_elems, tx_apo, burst, y_centers_pix, muteN, elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix);

if generate_frame_pair
    [scan_lines_post, rf_post, rf_raw_post] = GenerateRFLines(Nx, Ny, dy, num_scan_lines, sound_speed_map_post, density_map_post, y_centers_local_m, focus_depth, x_src_m, ...
        c0, kgrid, DATA_CAST, num_elems, tx_apo, burst, y_centers_pix, muteN, elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix);
end
%% ------------------- Processing (as in 3-D example) -------------------
tgc_alpha = 0.4;
compression_ratio = 3;
scale_factor = 2;
[scan_lines_pre, scan_lines_fund_pre, r] = ProcessRF(scan_lines_pre, num_scan_lines, tgc_alpha, burstN, kgrid, ...
    tone_burst_freq, compression_ratio, scale_factor,c0);
if generate_frame_pair
    [scan_lines_post, scan_lines_fund_post, r] = ProcessRF(scan_lines_post, num_scan_lines, tgc_alpha, burstN, kgrid, ...
        tone_burst_freq, compression_ratio, scale_factor,c0);
end
disp('test')
if visualize
    disp('test2')
    if generate_frame_pair
        % Phantom (truncate edges to match scanning sweep)
        figure;
        subplot(122);
        % B-mode (fundamental / harmonic)
        horz_axis = (0:length(scan_lines_fund_pre(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund_pre.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode - Window');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
        
        % B-mode (fundamental / harmonic)
        horz_axis = (0:length(scan_lines_fund_post(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        subplot(121)
        imagesc(horz_axis, r*1e3, scan_lines_fund_post.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
    else
                % B-mode (fundamental / harmonic)
                disp('test3')
        figure;
        horz_axis = (0:length(scan_lines_fund_pre(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund_pre.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
    end
else
    if generate_frame_pair
        % Phantom (truncate edges to match scanning sweep)
        f = figure('Visible','off');
        subplot(121);
        % B-mode (fundamental / harmonic)
        horz_axis = (0:length(scan_lines_fund_pre(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund_pre.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
        
        subplot(122);
        % B-mode (fundamental / harmonic)
        horz_axis = (0:length(scan_lines_fund_post(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund_post.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
        
    else
        f = figure('Visible','off');
        % B-mode (fundamental / harmonic)
        horz_axis = (0:length(scan_lines_fund_pre(:,1))-1)*elem_width_gdpts*dy/scale_factor*1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund_pre.'); axis image; colormap(gray);
        set(gca,'YLim',[5,40]); title('B-mode');
        xlabel('Lateral [mm]'); ylabel('Depth [mm]');
    end
        
    file = strcat('Result_',num2str(i),'.png');
    exportgraphics(f, file, 'Resolution',300);
end

    if (~exist("scan_lines_post","var"))
        scan_lines_post = zeros(25,25);
        scan_lines_fund_post = zeros(25,25);
        rf_post = zeros(25,25);
        rf_raw_post = zeros(25,25);
    end