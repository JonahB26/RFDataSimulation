% Multi-frequency comparison of WINDOWED vs FULL k-Wave simulations
% Tests windowing technique at 3, 5, 7.5, 10, and 12 MHz
% Properly refines grid spacing for each frequency to satisfy spatial sampling
% Saves comparison metrics to CSV file for analysis

%% ------------------- User inputs -------------------
clc; clear;
import kwave.*
filename = 'FEA_15_Resize.mat';  % <- set to your FEA output .mat
rng(42);                          % reproducibility

% Probe frequencies to test (Hz)
test_frequencies = [3e6, 5e6, 7.5e6, 10e6, 12e6];
% test_frequencies = [10e6, 12e6];

num_frequencies = length(test_frequencies);

% Output CSV filename
output_csv = 'windowing_comparison_results.csv';

%% ------------------- Controls (shared across all frequencies) ----------
DATA_CAST      = 'single';   % 'single' or 'gpuArray-single'
pml_x_size     = 20;
pml_y_size     = 10;

x_span         = 40e-3;      % depth FOV (meters) - FIXED physical size
c0             = 1540;       % m/s
rho0           = 1000;       % kg/m^3

ppw            = 4;          % points per wavelength (3-6 typical, 4 is reasonable)

num_elems        = 64;
elem_width_gdpts = 2;        % NOTE: this is in grid points, will scale with dx
kerf_gdpts       = 0;

tone_burst_cycles = 4;
source_strength   = 1e6;
guard_m           = 2e-3;
steer_deg         = 0;
focus_depth       = 20e-3;
num_scan_lines    = 96;
kNt               = 2500;

bg_mean = 1;  bg_std = 0.008;
halo_pix = 2;

%% ------------------- Initialize results table -------------------
results = table('Size', [num_frequencies, 15], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Frequency_MHz', 'dx_mm', 'Nx_eff', 'Ny_eff_win', 'Ny_eff_full', ...
                      'Time_Windowed_s', 'Time_Full_s', 'Speedup', 'NCC_Global', ...
                      'RMSE_Absolute', 'RMSE_vs_RMS_pct', 'RMSE_vs_Peak_pct', ...
                      'PSNR_dB', 'NCC_Mean', 'NCC_Std'});

%% ------------------- Load FEA data once -------------------
output = load(filename).output;
mask_img     = logical(output.images.tumor_mask);
cooper_mask  = logical(output.images.cooper_mask);
axial_disp   = output.disps.axial_disp   / 1000;  % [m]
lateral_disp = output.disps.lateral_disp / 1000;  % [m]

if isfield(output,'tumor_info') && isfield(output.tumor_info,'label') && strcmp(output.tumor_info.label,'malignant')
    inc_mean = 1570; inc_std = 75;
else
    inc_mean = 1600; inc_std = 75;
end

%% ------------------- Loop over frequencies ----------------------------
for freq_idx = 1:num_frequencies
    tone_burst_freq = test_frequencies(freq_idx);
    freq_mhz = tone_burst_freq / 1e6;
    
    fprintf('\n========================================\n');
    fprintf('Testing Frequency: %.1f MHz (%d/%d)\n', freq_mhz, freq_idx, num_frequencies);
    fprintf('========================================\n');
    
    %% Calculate required grid spacing for this frequency
    dx_req = c0 / (ppw * tone_burst_freq);
    Nx_int = ceil(x_span / dx_req);
    Nx_eff = Nx_int + 2*pml_x_size;
    
    dx = x_span / Nx_int;  % actual dx used
    dy = dx;               % keep square pixels
    
    f_max_spatial = c0 / (2*dx);
    fprintf('Frequency: %.1f MHz\n', freq_mhz);
    fprintf('Required dx: %.4f mm (PPW=%d)\n', dx_req*1e3, ppw);
    fprintf('Actual dx: %.4f mm\n', dx*1e3);
    fprintf('Nx_eff: %d (interior: %d)\n', Nx_eff, Nx_int);
    fprintf('k-Wave f_max: %.2f MHz\n', f_max_spatial/1e6);
    fprintf('Safety margin: %.2fx\n\n', f_max_spatial/tone_burst_freq);
    
    %% Build grids for this frequency
    aperture_pix        = num_elems * elem_width_gdpts;
    Ny_window_interior  = aperture_pix + 2*halo_pix;
    Ny_eff_win          = Ny_window_interior + 2*pml_y_size;
    
    % Windowed grid
    [Nx_win, Ny_win, dx_win, dy_win, kgrid_win] = GenerateKGrid(Nx_eff, Ny_eff_win, x_span, pml_x_size, pml_y_size, c0);
    kgrid_win.Nt = kNt;
    
    %% Generate linear array for this grid
    [x_src_pix, x_src_m, y_centers_pix_local, y_centers_local_m, tx_apo] = ...
        GenerateLinearArray(pml_x_size, dx, dy, Ny_win, num_elems, elem_width_gdpts, kerf_gdpts);
    
    %% Generate phantom for this grid resolution
    [sound_speed_map_pre, density_map_pre, Nx_tot, Ny_tot] = GenerateMedium(...
        Nx_win, Ny_win, dx, dy, num_scan_lines, elem_width_gdpts, ...
        bg_mean, bg_std, c0, rho0, ...
        mask_img, cooper_mask, axial_disp, lateral_disp, inc_mean, inc_std);
    
    % Full grid: lateral size must equal Ny_tot
    Ny_eff_full = Ny_tot + 2*pml_y_size;
    Nx_eff_full = Nx_tot + 2*pml_x_size;
    [Nx_full, Ny_full, dx_full, dy_full, kgrid_full] = GenerateKGrid(Nx_eff_full, Ny_eff_full, x_span, pml_x_size, pml_y_size, c0);
    kgrid_full.Nt = kNt;
    
    fprintf('WIN grid: Nx=%d, Ny=%d\n', kgrid_win.Nx, kgrid_win.Ny);
    fprintf('FULL grid: Nx=%d, Ny=%d\n\n', kgrid_full.Nx, kgrid_full.Ny);
    
    %% Generate input signal for this frequency
    [burst, burstN, ~, muteN] = GenerateInputSignal(kgrid_full, c0, rho0, ...
                                                     source_strength, tone_burst_freq, ...
                                                     tone_burst_cycles, guard_m);
    
    %% WINDOWED RUN
    fprintf('Running WINDOWED simulation...\n');
    tic;
    scan_lines_W = GenerateRFLines_Windowed( ...
        Nx_win, Ny_win, dy, num_scan_lines, ...
        sound_speed_map_pre, density_map_pre, ...
        y_centers_local_m, focus_depth, x_src_m, ...
        c0, kgrid_win, DATA_CAST, ...
        num_elems, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix);
    tW = toc;
    fprintf('Windowed complete: %.3f s\n\n', tW);
    
    %% FULL (non-windowed) RUN
    fprintf('Running FULL simulation...\n');
    tic;
    scan_lines_F = GenerateRFLines_Full( ...
        Nx_full, Ny_full, dy_full, num_scan_lines, ...
        sound_speed_map_pre, density_map_pre, ...
        y_centers_local_m, focus_depth, x_src_m, ...
        c0, kgrid_full, DATA_CAST, ...
        num_elems, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix, Ny_tot);
    tF = toc;
    fprintf('Full complete: %.3f s\n\n', tF);
    
    %% Compute similarity metrics
    W = gather_if_needed(scan_lines_W, DATA_CAST);
    F = gather_if_needed(scan_lines_F, DATA_CAST);
    
    rho_global  = corr(W(:), F(:));
    rmse_global = sqrt(mean((W(:)-F(:)).^2));
    
    sig_rms  = rms(F(:));
    sig_peak = max(abs(F(:)));
    
    rel_rmse_rms_pct = 100 * rmse_global / max(sig_rms, eps);
    rel_rmse_peak_pct = 100 * rmse_global / max(sig_peak, eps);
    psnr_like_db = 20*log10( max(sig_peak,eps) / max(rmse_global,eps) );
    
    % Per-line correlation
    rho_lines = zeros(num_scan_lines,1);
    for li=1:num_scan_lines
        rho_lines(li) = corr(W(li,:).', F(li,:).');
    end
    
    %% Store results
    results.Frequency_MHz(freq_idx) = freq_mhz;
    results.dx_mm(freq_idx) = dx * 1e3;
    results.Nx_eff(freq_idx) = Nx_eff;
    results.Ny_eff_win(freq_idx) = Ny_eff_win;
    results.Ny_eff_full(freq_idx) = Ny_eff_full;
    results.Time_Windowed_s(freq_idx) = tW;
    results.Time_Full_s(freq_idx) = tF;
    results.Speedup(freq_idx) = tF / max(tW, eps);
    results.NCC_Global(freq_idx) = rho_global;
    results.RMSE_Absolute(freq_idx) = rmse_global;
    results.RMSE_vs_RMS_pct(freq_idx) = rel_rmse_rms_pct;
    results.RMSE_vs_Peak_pct(freq_idx) = rel_rmse_peak_pct;
    results.PSNR_dB(freq_idx) = psnr_like_db;
    results.NCC_Mean(freq_idx) = mean(rho_lines);
    results.NCC_Std(freq_idx) = std(rho_lines);
    
    %% Print summary
    fprintf('==== RESULTS FOR %.1f MHz ====\n', freq_mhz);
    fprintf('Grid: dx=%.4f mm, Nx=%d\n', dx*1e3, Nx_eff);
    fprintf('Time (Windowed):    %.3f s\n', tW);
    fprintf('Time (Full):        %.3f s\n', tF);
    fprintf('Speedup:            %.2fx\n', tF/max(tW,eps));
    fprintf('NCC (global):       %.6f\n', rho_global);
    fprintf('RMSE (absolute):    %.3f\n', rmse_global);
    fprintf('RMSE (vs RMS):      %.2f %%\n', rel_rmse_rms_pct);
    fprintf('RMSE (vs peak):     %.2f %%\n', rel_rmse_peak_pct);
    fprintf('PSNR-like:          %.2f dB\n', psnr_like_db);
    fprintf('NCC per-line: mean=%.6f, sd=%.6f\n\n', mean(rho_lines), std(rho_lines));
end

%% ------------------- Save results to CSV ----------------------------
writetable(results, output_csv);
fprintf('========================================\n');
fprintf('All simulations complete!\n');
fprintf('Results saved to: %s\n', output_csv);
fprintf('========================================\n\n');

%% ------------------- Display summary table --------------------------
disp('Summary of Results:');
disp(results);

%% ------------------- Optional: Create summary plots -----------------
figure('Position', [100, 100, 1400, 900]);

subplot(2,4,1);
plot(results.Frequency_MHz, results.Speedup, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('Speedup (Full/Windowed)');
title('Computational Speedup vs Frequency');
grid on;

subplot(2,4,2);
plot(results.Frequency_MHz, results.NCC_Global, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('Global NCC');
title('Normalized Cross-Correlation vs Frequency');
ylim([min(results.NCC_Global)-0.01, 1]);
grid on;

subplot(2,4,3);
plot(results.Frequency_MHz, results.RMSE_vs_Peak_pct, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('RMSE (% of peak)');
title('Relative RMSE vs Frequency');
grid on;

subplot(2,4,4);
plot(results.Frequency_MHz, results.PSNR_dB, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('PSNR (dB)');
title('PSNR vs Frequency');
grid on;

subplot(2,4,5);
plot(results.Frequency_MHz, results.Time_Full_s, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(results.Frequency_MHz, results.Time_Windowed_s, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('Time (s)');
title('Computation Time vs Frequency');
legend('Full', 'Windowed', 'Location', 'best');
grid on;

subplot(2,4,6);
yyaxis left;
plot(results.Frequency_MHz, results.dx_mm, '-o', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('dx (mm)');
yyaxis right;
plot(results.Frequency_MHz, results.Nx_eff, '-s', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('Nx_{eff}');
xlabel('Frequency (MHz)');
title('Grid Refinement vs Frequency');
grid on;

subplot(2,4,7);
errorbar(results.Frequency_MHz, results.NCC_Mean, results.NCC_Std, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('Per-line NCC (mean ± std)');
title('Per-line NCC vs Frequency');
grid on;

subplot(2,4,8);
plot(results.Frequency_MHz, results.RMSE_vs_RMS_pct, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Frequency (MHz)'); ylabel('RMSE (% of RMS)');
title('RMSE vs RMS Energy vs Frequency');
grid on;

sgtitle('Windowing Technique Performance Across Frequencies (Grid-Refined)');

%% ======================= Local functions ==============================

function [Nx, Ny, dx, dy, kgrid] = GenerateKGrid(Nx_eff, Ny_eff, x_span, pml_x_size, pml_y_size, c0)
Nx = Nx_eff - 2*pml_x_size;
Ny = Ny_eff - 2*pml_y_size;
dx = x_span / Nx;
dy = dx;
kgrid = makeGrid(Nx, dx, Ny, dy);
t_end = (Nx*dx) * 2.2 / c0;
kgrid.t_array = makeTime(kgrid, c0, [], t_end);
end

function [burst, burstN, guardN, muteN] = GenerateInputSignal(kgrid, c0, rho0, source_strength, tone_burst_freq, tone_burst_cycles, guard_m)
burst = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);
burst = (source_strength / (c0 * rho0)) * burst;
burstN = numel(burst);
guardN  = round((2*guard_m/c0) / kgrid.dt);
muteN   = min(kgrid.Nt, burstN*2 + guardN);
end

function [x_src_pix, x_src_m, y_centers_pix, y_centers_local_m, tx_apo] = GenerateLinearArray(pml_x_size, dx, dy, Ny, num_elems, elem_width_gdpts, kerf_gdpts)
x_src_pix = pml_x_size + 6;
x_src_m   = (x_src_pix-1)*dx;

ap_width  = num_elems*elem_width_gdpts + (num_elems-1)*kerf_gdpts;
y0_pix    = floor((Ny - ap_width)/2) + 1;
y_centers_pix = y0_pix + (elem_width_gdpts+kerf_gdpts)*(0:num_elems-1) ...
                      + floor((elem_width_gdpts-1)/2);
y_centers_local_m = (y_centers_pix-1) * dy;
tx_apo = hann(num_elems).';
end

function [sound_speed_map_pre, density_map_pre, Nx_tot, Ny_tot] = GenerateMedium(...
    Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
    bg_mean, bg_std, c0, rho0, ...
    mask_img, cooper_mask, axial_disp, lateral_disp, inc_mean, inc_std)

% Full phantom wide enough to slide laterally
Nx_tot = Nx;
Ny_tot = Ny + num_scan_lines * elem_width_gdpts;
noise = randn([Nx_tot, Ny_tot]);

background_map = bg_mean + bg_std * noise;
sound_speed_map_pre = c0   * background_map;
density_map_pre     = rho0 * background_map;

% Resize displacements and masks to current grid resolution
if ~isequal(size(axial_disp), [Nx_tot, Ny_tot])
    axial_disp   = imresize(axial_disp,   [Nx_tot, Ny_tot], 'bicubic');
    lateral_disp = imresize(lateral_disp, [Nx_tot, Ny_tot], 'bicubic');
end
if ~isequal(size(mask_img), [Nx_tot, Ny_tot])
    mask_img = imresize(mask_img, [Nx_tot, Ny_tot], 'nearest');
end
if ~isequal(size(cooper_mask), [Nx_tot, Ny_tot])
    cooper_mask = imresize(cooper_mask, [Nx_tot, Ny_tot], 'nearest');
end

% Inclusion contrast
c_inc   = inc_mean + inc_std * noise;
c_inc  = min(max(c_inc, 1500), 1620);
rho_inc = c_inc / 1.5;
sound_speed_map_pre(mask_img) = c_inc(mask_img);
density_map_pre(mask_img)     = rho_inc(mask_img);

% Cooper's ligaments
c_coop   = 1610 + 25 * noise;
c_coop  = min(max(c_coop, 1550), 1650);
rho_coop = c_coop / 1.5;
sound_speed_map_pre(cooper_mask) = c_coop(cooper_mask);
density_map_pre(cooper_mask)     = rho_coop(cooper_mask);
end

function [scan_lines] = GenerateRFLines_Windowed( ...
    Nx, Ny, dy, num_scan_lines, sound_speed_map, density_map, ...
    y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
    num_elems, tx_apo, burst, y_centers_pix, muteN, elem_width_gdpts, ...
    pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix)

if ~strcmp(DATA_CAST,'gpuArray-single')
    scan_lines = zeros(num_scan_lines, kgrid.Nt, DATA_CAST);
else
    scan_lines = gpuArray(single(zeros(num_scan_lines,kgrid.Nt)));
end

input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], ...
              'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

medium.alpha_coeff = 0.75; medium.alpha_power = 1.5; medium.BonA = 6;

medium_position = 1;
h = waitbar(0,'STARTING');
for li = 1:num_scan_lines
    medium.sound_speed = sound_speed_map(:, medium_position:medium_position+Ny-1);
    medium.density     = density_map(:,   medium_position:medium_position+Ny-1);

    y0_m = (medium_position - 1) * dy;
    y_centers_global_m = y0_m + y_centers_local_m;

    y_focus_m = mean(y_centers_global_m) + focus_depth * tand(steer_deg);

    u_mask = false(Nx, Ny);
    u_mask(x_src_pix, y_centers_pix) = true;

    r_e = sqrt( (focus_depth - x_src_m).^2 + (y_centers_global_m - y_focus_m).^2 );
    tau = (max(r_e) - r_e) / c0;
    
    if ~strcmp(DATA_CAST,'gpuArray-single')
        drive = zeros(num_elems, kgrid.Nt, DATA_CAST);
    else 
        drive = gpuArray(single(zeros(num_elems,kgrid.Nt)));
    end
    for e = 1:num_elems
        idx0 = 1 + round(tau(e) / kgrid.dt);
        if idx0 <= kgrid.Nt
            nAvail = min(burstN, kgrid.Nt - idx0 + 1);
            drive(e, idx0:idx0+nAvail-1) = tx_apo(e) * burst(1:nAvail);
        end
    end

    source = struct(); source.u_mask = u_mask; source.ux = drive;
    sensor = struct(); sensor.mask = u_mask; sensor.record = {'p'};
    
    raw = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    rf_raw = raw.p;
    rf_raw(:,1:muteN) = 0;
    rf = rf_raw - mean(rf_raw,2);

    num_rx = size(rf,1);
    rx_apo = ones(num_rx,1,'like',rf);

    z_m = (0:kgrid.Nt-1) * kgrid.dt * c0/2;
    if strcmp(DATA_CAST,'gpuArray-single')
        z_m = gpuArray(cast(z_m, 'like', rf));
    else
        z_m = cast(z_m, 'like', rf);
    end
    z_m_row = reshape(z_m, 1, []);

    if strcmp(DATA_CAST,'gpuArray-single')
        ycg = gpuArray(cast(y_centers_global_m(:), 'like', rf));
    else
        ycg = cast(y_centers_global_m(:),'like',rf);
    end

    tx_yc = mean(y_centers_global_m);
    tx_xc = x_src_m;
    y_line_m = y_focus_m;

    r_tx = sqrt( (z_m_row - tx_xc).^2 + (y_line_m - tx_yc).^2 );

    Z = repmat(z_m_row, num_rx, 1);
    Y = repmat(ycg, 1, kgrid.Nt);
    r_rx = sqrt( (Z - x_src_m).^2 + (Y - y_line_m).^2 );

    tof  = (bsxfun(@plus, r_tx, r_rx)) / c0;
    idxf = 1 + tof / kgrid.dt;

    i0 = max(1, min(floor(idxf), kgrid.Nt-1));
    i1 = i0 + 1;
    a  = idxf - i0;

    if strcmp(DATA_CAST,'gpuArray-single')
        rowsMat = gpuArray(repmat((1:num_rx).', 1, kgrid.Nt));
    else
        rowsMat = repmat((1:num_rx).', 1, kgrid.Nt);
    end

    v0 = rf(sub2ind([num_rx, kgrid.Nt], rowsMat, i0));
    v1 = rf(sub2ind([num_rx, kgrid.Nt], rowsMat, i1));
    vals = (1 - a).*v0 + a.*v1;

    scan = (rx_apo.' * vals);
    scan(1:muteN) = 0;

    scan_lines(li,:) = gather_if_needed(scan, DATA_CAST);

    medium_position = medium_position + elem_width_gdpts;
    frac = li/num_scan_lines;
    waitbar(frac,h,sprintf('Done %d | %d Windowed Wave Sim',li,num_scan_lines));
end
end

function [scan_lines] = GenerateRFLines_Full( ...
    Nx, Ny, dy, num_scan_lines, sound_speed_map, density_map, ...
    y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
    num_elems, tx_apo, burst, y_centers_pix_local, muteN, ...
    elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix, Ny_tot)

assert(size(sound_speed_map,2)==Ny_tot && kgrid.Ny==Ny_tot, ...
    'kgrid.Ny and medium Ny must match full phantom width (Ny_tot).');

if ~strcmp(DATA_CAST,'gpuArray-single')
    scan_lines = zeros(num_scan_lines, kgrid.Nt, DATA_CAST);
else
    scan_lines = gpuArray(single(zeros(num_scan_lines,kgrid.Nt)));
end

input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], ...
              'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

medium.alpha_coeff = 0.75; medium.alpha_power = 1.5; medium.BonA = 6;

medium.sound_speed = sound_speed_map;
medium.density     = density_map;

h = waitbar(0,'STARTING');
for li = 1:num_scan_lines
    lateral_shift_pix = (li-1) * elem_width_gdpts;

    y_centers_pix_line = y_centers_pix_local + lateral_shift_pix;
    y_centers_pix_line = min(y_centers_pix_line, Ny_tot);
    y_centers_global_m = (y_centers_pix_line - 1) * dy;

    y_focus_m = mean(y_centers_global_m) + focus_depth * tand(steer_deg);

    u_mask = false(Nx, Ny_tot);
    u_mask(x_src_pix, y_centers_pix_line) = true;

    r_e = sqrt( (focus_depth - x_src_m).^2 + (y_centers_global_m - y_focus_m).^2 );
    tau = (max(r_e) - r_e) / c0;
    
    if ~strcmp(DATA_CAST,'gpuArray-single')
        drive = zeros(num_elems, kgrid.Nt, DATA_CAST);
    else
        drive = gpuArray(single(zeros(num_elems,kgrid.Nt)));
    end
    for e = 1:num_elems
        idx0 = 1 + round(tau(e) / kgrid.dt);
        if idx0 <= kgrid.Nt
            nAvail = min(burstN, kgrid.Nt - idx0 + 1);
            drive(e, idx0:idx0+nAvail-1) = tx_apo(e) * burst(1:nAvail);
        end
    end

    source = struct(); source.u_mask = u_mask; source.ux = drive;
    sensor = struct(); sensor.mask = u_mask; sensor.record = {'p'};
    
    raw = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    rf_raw = raw.p; rf_raw(:,1:muteN) = 0;
    rf = rf_raw - mean(rf_raw,2);

    num_rx = size(rf,1);
    rx_apo = ones(num_rx,1,'like',rf);

    z_m = (0:kgrid.Nt-1) * kgrid.dt * c0/2;
    if strcmp(DATA_CAST,'gpuArray-single')
        z_m = gpuArray(cast(z_m, 'like', rf));
    else
        z_m = cast(z_m,'like',rf);
    end
    z_m_row = reshape(z_m, 1, []);

    if strcmp(DATA_CAST,'gpuArray-single')
        ycg = gpuArray(cast(y_centers_global_m(:), 'like', rf));
    else
        ycg = cast(y_centers_global_m(:),'like',rf);
    end

    tx_yc = mean(y_centers_global_m);
    tx_xc = x_src_m;
    y_line_m = y_focus_m;

    r_tx = sqrt( (z_m_row - tx_xc).^2 + (y_line_m - tx_yc).^2 );

    Z = repmat(z_m_row, num_rx, 1);
    Y = repmat(ycg,       1,      kgrid.Nt);
    r_rx = sqrt( (Z - x_src_m).^2 + (Y - y_line_m).^2 );

    tof  = (bsxfun(@plus, r_tx, r_rx)) / c0;
    idxf = 1 + tof / kgrid.dt;

    i0 = max(1, min(floor(idxf), kgrid.Nt-1));
    i1 = i0 + 1;
    a  = idxf - i0;

    if strcmp(DATA_CAST,'gpuArray-single')
        rowsMat = gpuArray(repmat((1:num_rx).', 1, kgrid.Nt));
    else
        rowsMat = repmat((1:num_rx).',1,kgrid.Nt);
    end

    v0 = rf(sub2ind([num_rx, kgrid.Nt], rowsMat, i0));
    v1 = rf(sub2ind([num_rx, kgrid.Nt], rowsMat, i1));
    vals = (1 - a).*v0 + a.*v1;

    scan = (rx_apo.' * vals);
    scan(1:muteN) = 0;

    scan_lines(li,:) = gather_if_needed(scan, DATA_CAST);
    waitbar(li/num_scan_lines,h,sprintf('Done %d | %d Full Wave Sim',li,num_scan_lines));
end
end

function out = gather_if_needed(x, CAST)
if contains(CAST,'gpu'), out = gather(x); else, out = x; end
end