% k-Wave Elastography Simulator with Windowing Technique
% Generates PRE and POST compression RF data frames
% Compatible with Field II elastography pipeline
% Outputs: RF_Data_Pre, RF_Data_Post, Tstarts (same format as Field II)

%% ------------------- User inputs -------------------
clc; clear;
import kwave.*

rng(42);

% Output filenames
output_pre  = 'kwave_rf_pre.mat';
output_post = 'kwave_rf_post.mat';

%% ------------------- Simulation Controls -------------------
DATA_CAST      = 'single';   % 'single' or 'gpuArray-single'
USE_WINDOWING  = true;       % Set to false for full simulation

pml_x_size     = 20;
pml_y_size     = 10;
x_span         = 40e-3;      % 40mm depth
c0             = 1540;       % m/s
rho0           = 1000;       % kg/m^3

% Transducer parameters (matching Field II style)
tone_burst_freq   = 5e6;     % 5 MHz center frequency
tone_burst_cycles = 2;       % Shorter pulse for better axial resolution
source_strength   = 1e6;     % Pa

num_elems        = 128;      % Total elements (not all used if windowing)
num_active       = 64;       % Active aperture
elem_width_gdpts = 2;
kerf_gdpts       = 0;

steer_deg        = 0;
focus_depth      = 20e-3;    % 20mm transmit focus
guard_m          = 2e-3;
kNt              = 2500;

num_scan_lines   = 96;       % Number of A-lines

bg_mean = 1;  bg_std = 0.008;
halo_pix = 2;

%% ------------------- Calculate grid spacing -------------------
% Option 1: Use fixed Nx_eff (matches your original script)
% Nx_eff = 256;  % Fixed grid size
% Nx_int = Nx_eff - 2*pml_x_size;
% dx = x_span / Nx_int;
% dy = dx;

% fprintf('Grid spacing: dx = %.4f mm\n', dx*1e3);
% fprintf('Grid size: Nx_eff = %d\n', Nx_eff);

% Option 2: Use frequency-dependent refinement (uncomment to use)
ppw = 4;
dx_req = c0 / (ppw * tone_burst_freq);
Nx_int = ceil(x_span / dx_req);
Nx_eff = Nx_int + 2*pml_x_size;
dx = x_span / Nx_int;
dy = dx;
fprintf('Grid spacing: dx = %.4f mm (PPW=%d)\n', dx*1e3, ppw);
fprintf('Grid size: Nx_eff = %d\n', Nx_eff);

%% ------------------- Build initial grid for windowing -------------------
aperture_pix        = num_active * elem_width_gdpts;  % Use num_active, not num_elems!
Ny_window_interior  = aperture_pix + 2*halo_pix;
Ny_eff_win          = Ny_window_interior + 2*pml_y_size;

% Always create windowed grid first to get dx/dy
[Nx_win, Ny_win, dx, dy, kgrid_temp] = GenerateKGrid(Nx_eff, Ny_eff_win, x_span, pml_x_size, pml_y_size, c0);
fprintf('Windowed grid would be: Nx=%d, Ny=%d\n', Nx_win, Ny_win);

%% ------------------- Synthetic phantom + synthetic displacement controls -------------------
% These parameters are intentionally set so PRE vs POST should differ.
% If PRE/POST still look identical, increase eps_axial to 0.05 or bg_std to ~0.02.
inc_mean     = 1600;     % inclusion sound speed mean (m/s)
inc_std      = 50;       % inclusion sound speed std (m/s)
eps_axial    = 0.03;     % 3% axial compression (increase for more decorrelation)
nu           = 0.45;     % lateral coupling (Poisson-like), set to 0 for purely axial
stiff_scale  = 0.50;     % inclusion moves less (stiffer) -> reduced displacement in inclusion

%% ------------------- Generate phantom and finalize grid -------------------
if USE_WINDOWING
    fprintf('Using WINDOWING mode\n');

    % Use windowed dimensions for phantom
    Nx = Nx_win;
    Ny = Ny_win;
    kgrid = kgrid_temp;
    kgrid.Nt = kNt;

    % Generate array for windowed grid
    [x_src_pix, x_src_m, y_centers_pix_local, y_centers_local_m, tx_apo] = ...
        GenerateLinearArray(pml_x_size, dx, dy, Ny, num_active, elem_width_gdpts, kerf_gdpts);

    % Build synthetic masks + displacement directly on [Nx_tot, Ny_tot]
    [mask_img, cooper_mask, axial_disp, lateral_disp] = ...
        GenerateSyntheticPhantomAndDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                                focus_depth, eps_axial, nu, stiff_scale);

    fprintf('Synthetic disp: max|axial| = %.3e m (%.2f px)\n', max(abs(axial_disp(:))), max(abs(axial_disp(:)))/dx);
    fprintf('Synthetic disp: max|lat|   = %.3e m (%.2f px)\n', max(abs(lateral_disp(:))), max(abs(lateral_disp(:)))/dy);

    % Generate phantom with windowed dimensions
    [sound_speed_map_pre, density_map_pre, sound_speed_map_post, density_map_post, Nx_tot, Ny_tot] = ...
        GenerateMediumWithDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                       bg_mean, bg_std, c0, rho0, ...
                                       mask_img, cooper_mask, axial_disp, lateral_disp, inc_mean, inc_std);

    figure('Name','Sound speed maps (PRE vs POST)','Color','w');

    subplot(1,3,1)
    imagesc(sound_speed_map_pre); axis image; colorbar;
    title('PRE sound speed'); xlabel('Lateral'); ylabel('Depth');
    
    subplot(1,3,2)
    imagesc(sound_speed_map_post); axis image; colorbar;
    title('POST sound speed');
    
    subplot(1,3,3)
    imagesc(sound_speed_map_post - sound_speed_map_pre);
    axis image; colorbar;
    title('POST − PRE difference');


    fprintf('Windowed phantom: Nx_tot=%d, Ny_tot=%d\n', Nx_tot, Ny_tot);
    fprintf('Grid used: Nx=%d, Ny=%d (slides across phantom)\n', Nx, Ny);

else
    fprintf('Using FULL simulation (no windowing)\n');

    % First generate full-width phantom to know Ny_tot
    Nx = Nx_win;
    Ny = Ny_win;

    % Build synthetic masks + displacement directly on [Nx_tot, Ny_tot] for the FULL phantom width
    [mask_img, cooper_mask, axial_disp, lateral_disp] = ...
        GenerateSyntheticPhantomAndDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                                focus_depth, eps_axial, nu, stiff_scale);

    fprintf('Synthetic disp: max|axial| = %.3e m (%.2f px)\n', max(abs(axial_disp(:))), max(abs(axial_disp(:)))/dx);
    fprintf('Synthetic disp: max|lat|   = %.3e m (%.2f px)\n', max(abs(lateral_disp(:))), max(abs(lateral_disp(:)))/dy);

    [sound_speed_map_pre, density_map_pre, sound_speed_map_post, density_map_post, Nx_tot, Ny_tot] = ...
        GenerateMediumWithDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                       bg_mean, bg_std, c0, rho0, ...
                                       mask_img, cooper_mask, axial_disp, lateral_disp, inc_mean, inc_std);

    % Now rebuild grid to match full phantom width
    Ny_eff_full = Ny_tot + 2*pml_y_size;
    Nx_eff_full = Nx_tot + 2*pml_x_size;
    [Nx, Ny, dx, dy, kgrid] = GenerateKGrid(Nx_eff_full, Ny_eff_full, x_span, pml_x_size, pml_y_size, c0);
    kgrid.Nt = kNt;

    fprintf('Full phantom: Nx_tot=%d, Ny_tot=%d\n', Nx_tot, Ny_tot);
    fprintf('Full grid: Nx=%d, Ny=%d\n', Nx, Ny);

    % Regenerate array for full grid
    [x_src_pix, x_src_m, y_centers_pix_local, y_centers_local_m, tx_apo] = ...
        GenerateLinearArray(pml_x_size, dx, dy, Ny, num_active, elem_width_gdpts, kerf_gdpts);
end

% Optional sanity: confirm POST differs from PRE
fprintf('||c_post - c_pre|| / ||c_pre|| = %.3e\n', ...
    norm(sound_speed_map_post(:)-sound_speed_map_pre(:))/norm(sound_speed_map_pre(:)));

%% ------------------- Generate input signal -------------------
[burst, burstN, ~, muteN] = GenerateInputSignal(kgrid, c0, rho0, source_strength, tone_burst_freq, tone_burst_cycles, guard_m);

%% ================= PRE-COMPRESSION SIMULATION =================
fprintf('\n========================================\n');
fprintf('Running PRE-compression simulation...\n');
fprintf('========================================\n');
tic;

if USE_WINDOWING
    scan_lines_pre = GenerateRFLines_Windowed(Nx, Ny, dy, num_scan_lines, ...
        sound_speed_map_pre, density_map_pre, ...
        y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
        num_active, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix);
else
    scan_lines_pre = GenerateRFLines_Full(Nx, Ny, dy, num_scan_lines, ...
        sound_speed_map_pre, density_map_pre, ...
        y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
        num_active, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix, Ny_tot);
end

t_pre = toc;
fprintf('PRE-compression complete: %.2f s\n', t_pre);

%% ================= POST-COMPRESSION SIMULATION =================
fprintf('\n========================================\n');
fprintf('Running POST-compression simulation...\n');
fprintf('========================================\n');
tic;

if USE_WINDOWING
    scan_lines_post = GenerateRFLines_Windowed(Nx, Ny, dy, num_scan_lines, ...
        sound_speed_map_post, density_map_post, ...
        y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
        num_active, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix);
else
    scan_lines_post = GenerateRFLines_Full(Nx, Ny, dy, num_scan_lines, ...
        sound_speed_map_post, density_map_post, ...
        y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
        num_active, tx_apo, burst, y_centers_pix_local, muteN, ...
        elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix, Ny_tot);
end

t_post = toc;
fprintf('POST-compression complete: %.2f s\n', t_post);

li = round(num_scan_lines/2); % center scan line

figure('Name','RF comparison','Color','w');

subplot(2,1,1)
plot(scan_lines_pre(li,:), 'k');
title(sprintf('PRE RF – line %d', li));
xlabel('Time samples'); ylabel('Amplitude');

subplot(2,1,2)
plot(scan_lines_post(li,:), 'r');
title(sprintf('POST RF – line %d', li));
xlabel('Time samples'); ylabel('Amplitude');

env_pre  = abs(hilbert(scan_lines_pre.')).';
env_post = abs(hilbert(scan_lines_post.')).';

env_pre  = env_pre  ./ max(env_pre(:));
env_post = env_post ./ max(env_post(:));

figure('Name','B-mode PRE vs POST','Color','w');

subplot(1,2,1)
imagesc(20*log10(env_pre + eps), [-60 0]);
colormap gray; axis image;
title('PRE B-mode'); xlabel('Lateral'); ylabel('Depth');

subplot(1,2,2)
imagesc(20*log10(env_post + eps), [-60 0]);
colormap gray; axis image;
title('POST B-mode');

rf_diff = norm(scan_lines_post(:) - scan_lines_pre(:)) / ...
          norm(scan_lines_pre(:));
fprintf('Relative RF difference = %.3e\n', rf_diff);
%% ================= Convert to Field II format =================
fprintf('\nConverting to Field II format...\n');

RF_Data_Pre  = cell(num_scan_lines, 1);
RF_Data_Post = cell(num_scan_lines, 1);
Tstarts      = zeros(num_scan_lines, 1);

scan_lines_pre  = gather_if_needed(scan_lines_pre,  DATA_CAST);
scan_lines_post = gather_if_needed(scan_lines_post, DATA_CAST);

for li = 1:num_scan_lines
    RF_Data_Pre{li}  = scan_lines_pre(li,:).';
    RF_Data_Post{li} = scan_lines_post(li,:).';
    Tstarts(li)      = 0; % if you have a consistent start time, place it here
end

save(output_pre,  'RF_Data_Pre',  'Tstarts', '-v7.3');
save(output_post, 'RF_Data_Post', 'Tstarts', '-v7.3');

fprintf('\n========================================\n');
fprintf('Elastography simulation complete!\n');
fprintf('Total time: %.2f s\n', t_pre + t_post);
fprintf('========================================\n');

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

function [mask_img, cooper_mask, axial_disp, lateral_disp] = ...
    GenerateSyntheticPhantomAndDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                            focus_depth, eps_axial, nu, stiff_scale)
% Build on the full phantom size used by GenerateMediumWithDisplacement.
Nx_tot = Nx;
Ny_tot = Ny + num_scan_lines * elem_width_gdpts;

[xg, yg] = ndgrid((0:Nx_tot-1)*dx, (0:Ny_tot-1)*dy);

% Inclusion centered near focus depth
x0 = focus_depth;
y0 = (Ny_tot*dy)/2;
R  = 4e-3;

mask_img = ((xg - x0).^2 + (yg - y0).^2) <= R^2;

% Simple "ligament-like" diagonals
cooper_mask = false(Nx_tot, Ny_tot);
num_lines = 5;
for k = 1:num_lines
    y_line = round((k/(num_lines+1)) * Ny_tot);
    for ix = 1:Nx_tot
        iy = y_line + round(0.10 * (ix - Nx_tot/3));
        if iy >= 1 && iy <= Ny_tot
            cooper_mask(ix, iy) = true;
        end
    end
end
cooper_mask = imdilate(cooper_mask, strel('disk', 1));

% Axial compression increasing with depth: u_x = -eps * x
axial_disp = -eps_axial * xg;

% Stiffer inclusion: reduced displacement magnitude
axial_disp(mask_img) = stiff_scale * axial_disp(mask_img);

% Lateral motion (Poisson-like expansion about centerline)
lateral_disp = nu * eps_axial * (yg - y0);

% Smooth to avoid high-gradient interpolation artifacts
axial_disp   = imgaussfilt(axial_disp,   1);
lateral_disp = imgaussfilt(lateral_disp, 1);
end

function [sound_speed_map_pre, density_map_pre, sound_speed_map_post, density_map_post, Nx_tot, Ny_tot] = ...
    GenerateMediumWithDisplacement(Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, ...
                                   bg_mean, bg_std, c0, rho0, ...
                                   mask_img, cooper_mask, axial_disp, lateral_disp, inc_mean, inc_std)

Nx_tot = Nx;
Ny_tot = Ny + num_scan_lines * elem_width_gdpts;
noise = randn([Nx_tot, Ny_tot]);

background_map = bg_mean + bg_std * noise;
sound_speed_map_pre = c0   * background_map;
density_map_pre     = rho0 * background_map;

% Resize displacements and masks if needed (kept for robustness)
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
c_inc   = min(max(c_inc, 1500), 1620);
rho_inc = c_inc / 1.5;
sound_speed_map_pre(mask_img) = c_inc(mask_img);
density_map_pre(mask_img)     = rho_inc(mask_img);

% Cooper's ligaments
c_coop   = 1610 + 25 * noise;
c_coop   = min(max(c_coop, 1550), 1650);
rho_coop = c_coop / 1.5;
sound_speed_map_pre(cooper_mask) = c_coop(cooper_mask);
density_map_pre(cooper_mask)     = rho_coop(cooper_mask);

% Apply deformation to create POST-compression maps
[xg, yg] = ndgrid((0:Nx_tot-1)*dx, (0:Ny_tot-1)*dy);
x_src = xg - axial_disp;
y_src = yg - lateral_disp;
x_src = min(max(x_src, 0), (Nx_tot-1)*dx);
y_src = min(max(y_src, 0), (Ny_tot-1)*dy);

Fc_pre   = griddedInterpolant(xg, yg, sound_speed_map_pre, 'linear', 'nearest');
Frho_pre = griddedInterpolant(xg, yg, density_map_pre,     'linear', 'nearest');

sound_speed_map_post = Fc_pre(x_src, y_src);
density_map_post     = Frho_pre(x_src, y_src);
end

function [scan_lines] = GenerateRFLines_Windowed(Nx, Ny, dy, num_scan_lines, sound_speed_map, density_map, ...
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

    scan = BeamformDAS(rf, kgrid, y_centers_global_m, y_focus_m, x_src_m, c0, DATA_CAST);
    scan(1:muteN) = 0;

    scan_lines(li,:) = gather_if_needed(scan, DATA_CAST);
    medium_position = medium_position + elem_width_gdpts;
end
end

function [scan_lines] = GenerateRFLines_Full(Nx, Ny, dy, num_scan_lines, sound_speed_map, density_map, ...
    y_centers_local_m, focus_depth, x_src_m, c0, kgrid, DATA_CAST, ...
    num_elems, tx_apo, burst, y_centers_pix_local, muteN, ...
    elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix, Ny_tot)

if ~strcmp(DATA_CAST,'gpuArray-single')
    scan_lines = zeros(num_scan_lines, kgrid.Nt, DATA_CAST);
else
    scan_lines = gpuArray(single(zeros(num_scan_lines,kgrid.Nt)));
end

input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], ...
              'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

medium.alpha_coeff = 0.75; medium.alpha_power = 1.5; medium.BonA = 6;

% In full mode, the phantom spans Ny_tot, but we still simulate using the full Ny window
% (i.e., no sliding); just use the first Ny columns (or center if you prefer).
% Here: use the centered Ny block.
start_col = max(1, floor((Ny_tot - Ny)/2) + 1);
end_col   = start_col + Ny - 1;

medium.sound_speed = sound_speed_map(:, start_col:end_col);
medium.density     = density_map(:,   start_col:end_col);

y0_m = (start_col - 1) * dy;
y_centers_global_m = y0_m + y_centers_local_m;
y_focus_m = mean(y_centers_global_m) + focus_depth * tand(steer_deg);

u_mask = false(Nx, Ny);
u_mask(x_src_pix, y_centers_pix_local) = true;

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

scan = BeamformDAS(rf, kgrid, y_centers_global_m, y_focus_m, x_src_m, c0, DATA_CAST);
scan(1:muteN) = 0;

% For full mode we return the same beamformed line for each "scan line" index,
% because in full mode there is no sliding medium. If you want true lateral scanning
% in full mode, you would step the aperture laterally and re-run; that is separate.
for li = 1:num_scan_lines
    scan_lines(li,:) = gather_if_needed(scan, DATA_CAST);
end
end

function out = gather_if_needed(x, DATA_CAST)
if strcmp(DATA_CAST,'gpuArray-single')
    out = gather(x);
else
    out = x;
end
end

function scan = BeamformDAS(rf, kgrid, y_centers_global_m, y_focus_m, x_src_m, c0, DATA_CAST)
% Simple delay-and-sum beamformer (kept as-is in your pipeline)
num_elems = size(rf,1);
Nt = size(rf,2);

t = (0:Nt-1) * kgrid.dt;

% Range from each element to focus point
r = sqrt( (x_src_m).^2 + (y_centers_global_m - y_focus_m).^2 );
tau = r / c0;

if ~strcmp(DATA_CAST,'gpuArray-single')
    scan = zeros(1, Nt, DATA_CAST);
else
    scan = gpuArray(single(zeros(1,Nt)));
end

for e = 1:num_elems
    te = t - tau(e);
    % linear interpolation in time
    scan = scan + interp1(t, rf(e,:), te, 'linear', 0);
end

scan = squeeze(scan);
end

