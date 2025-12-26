clc;clear; import kwave.*;
run_simulation = true; use_gpu_array = true; 

%% -------------------------- Parameters ---------------------------------
c0        = 1540;              % m/s
rho0      = 1000;              % kg/m^3
fc        = 5e6;               % Hz (transmit)
cycles    = 3;                 % tone-burst cycles
alpha_db  = 0.75;              % dB / (MHz^y cm)
alpha_pow = 1.5;               % y (power-law)

% Grid (compact compute tile swept laterally by updating medium maps)
dx = 50e-6; dy = 50e-6; dz = 50e-6;   % 0.05 mm voxels
Nx = 96; Ny = 128; Nz = 384;          % x (lateral across elements), y (elev), z (depth)
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

% Time step / record window (≈ 35 mm depth → ~45 µs round-trip)
cfl  = 0.3;
dt   = cfl * min([dx, dy, dz]) / c0;
tEnd = 4.5e-5;                        % 45 µs
kgrid.setTime(round(tEnd/dt), dt);

% Transducer (32 active elements; swept to form 96 scan lines)
number_scan_lines   = 96;
active_elements     = 32;             % number of TX/RX channels used per line
element_width_pts   = 2;              % in grid points (x)
element_length_pts  = Ny - 8;         % in grid points (y, elevation)
element_spacing_pts = 0;              % kerf in points (x)
focus_mm            = 15;             % TX/RX focus in mm (beam plane)

% File to save/reload RF
rf_file = 'example_us_bmode_scan_lines.mat';

%% --------------------- Build scattering phantom ------------------------
% Larger (swept) phantom maps: make enough lateral span for all 96 lines
tile_x_mm = Nx*dx*1e3;
sweep_extra_lines = number_scan_lines - active_elements;     % how far we slide
sweep_extra_pts   = sweep_extra_lines * element_width_pts;   % in x-points
Nx_full = Nx + sweep_extra_pts;                              % full phantom width

% Base maps
sound_speed_map = c0  * ones(Nx_full, Ny, Nz, 'single');
density_map     = rho0 * ones(Nx_full, Ny, Nz, 'single');

% Micro-scale speckle: Gaussian noise on c and rho
rng(1);
c_sigma   = 0.02;     % ±2% c variation
rho_sigma = 0.02;     % ±2% rho variation
sound_speed_map = sound_speed_map .* (1 + c_sigma   * randn(size(sound_speed_map), 'single'));
density_map     = density_map     .* (1 + rho_sigma * randn(size(density_map),     'single'));

% 3 spherical inclusions (co-aligned in elevation y)
% centers given in (x,z) mm relative to compute-tile center; radius mm
incs = [  0, 12, 3.0,  +0.05, +0.05;   % [x_mm, z_mm, R_mm, dC_frac, dR_frac]
         -4, 18, 2.5,  +0.06, +0.04;
         +5, 22, 1.8,  +0.08, +0.04 ];
[xm, ym, zm] = kgrid.getMesh();   % meters

% convert inclusion centers from mm to meters and map to the *tile* center
x0_tile = mean(xm(:,1,1));
z0_tile = min(zm(1,1,:));  % top is ~0, depth increases with z-index

for k = 1:size(incs,1)
    cx = x0_tile + incs(k,1)*1e-3;
    cz = z0_tile + incs(k,2)*1e-3;
    R  = incs(k,3)*1e-3;
    % sphere mask on the tile extents (use Ny/2 slice for elevation alignment)
    SM = ( (xm - cx).^2 + (zm - cz).^2 ) <= R^2;
    % apply to the *central* y-slice and then replicate in y to keep spheres co-aligned
    for yi = 1:Ny
        sound_speed_map(1:Nx, yi, 1:Nz) = sound_speed_map(1:Nx, yi, 1:Nz) .* (1 + incs(k,4)*single(SM));
        density_map(    1:Nx, yi, 1:Nz) = density_map(    1:Nx, yi, 1:Nz) .* (1 + incs(k,5)*single(SM));
    end
end

%% --------------------- Define transducer (source + sensor) -------------
transducer = kWaveTransducer(kgrid, struct( ...
    'number_elements',     active_elements, ...
    'element_width',       element_width_pts, ...
    'element_length',      element_length_pts, ...
    'element_spacing',     element_spacing_pts, ...
    'radius',              inf, ...               % linear array
    'position',            [Nx/2, Ny/2, 1], ...  % centered, at top surface
    'sound_speed',         c0, ...
    'focus_distance',      focus_mm/1e3, ...
    'elevation_focus_distance', inf, ...
    'steering_angle',      0, ...
    'transmit_apodization','Hanning', ...
    'receive_apodization', 'Hanning' ...
    ));

% Tone-burst drive (k-Wave helper builds per-element TX delays internally)
input_signal = toneBurst(1/kgrid.dt, fc, cycles, 'Envelope', 'Gaussian');
transducer.input_signal = input_signal;

% Medium absorption (k-Wave uses dB/(MHz^y cm))
medium.alpha_coeff = alpha_db;
medium.alpha_power = alpha_pow;

%% ---------------------- Run or reload RF -------------------------------
if run_simulation
    fprintf('Running 3D k-Wave simulation for %d scan lines...\n', number_scan_lines);

    % base maps for the *compute tile* (will be overwritten inside the loop)
    medium.sound_speed = sound_speed_map(1:Nx, :, :);
    medium.density     = density_map(    1:Nx, :, :);

    % solver args
    if use_gpu_array
        input_args = {'PlotSim', false, 'PMLAlpha', 2, 'DataCast', 'gpuArray-single'};
    else
        input_args = {'PlotSim', false, 'PMLAlpha', 2, 'DataCast', 'single'};
    end

    scan_lines = zeros(number_scan_lines, kgrid.Nt, 'single');
    medium_position = 1;   % x-index where the active aperture starts

    for sl = 1:number_scan_lines
        % load the current *tile* from the larger phantom by shifting x
        medium.sound_speed = sound_speed_map(medium_position:medium_position+Nx-1, :, :);
        medium.density     = density_map(    medium_position:medium_position+Nx-1, :, :);

        % run 3D simulation; transducer used for both source and sensor
        sensor_data = kspaceFirstOrder3D(kgrid, medium, transducer, transducer, input_args{:});

        % form a scan line using transducer's built-in beamformer
        scan_lines(sl, :) = transducer.scan_line(sensor_data);

        % advance medium window laterally by one element width
        medium_position = medium_position + transducer.element_width;
        fprintf('  line %3d/%3d done\n', sl, number_scan_lines);
    end

    % save RF (so you can reprocess without re-simulating)
    save(rf_file, 'scan_lines', 'kgrid', 'c0', 'rho0', 'fc', 'alpha_db', 'alpha_pow', ...
         'number_scan_lines', 'cycles', 'focus_mm', 'active_elements', '-v7.3');
else
    fprintf('Loading saved RF from %s...\n', rf_file);
    S = load(rf_file);
    scan_lines        = S.scan_lines;
    kgrid             = S.kgrid;
    c0                = S.c0;
    fc                = S.fc;
    alpha_db          = S.alpha_db;
    alpha_pow         = S.alpha_pow;
    number_scan_lines = S.number_scan_lines;
    focus_mm          = S.focus_mm;
    active_elements   = S.active_elements;
end

%% ------------------------- Processing steps ----------------------------
% (1) TGC (time-gain compensation) using tissue attenuation model)
t0 = numel(input_signal) * kgrid.dt / 2;             % middle of burst
r  = c0 * ((1:kgrid.Nt) * kgrid.dt / 2 - t0);        % meters, two-way
tgc_alpha_db_cm = alpha_db * (fc * 1e-6)^alpha_pow;  % dB/cm
tgc_alpha_np_m  = (tgc_alpha_db_cm / 8.686) * 100;   % Np/m
tgc = exp(tgc_alpha_np_m * 2 * r);                   % round-trip distance
scan_lines = bsxfun(@times, tgc, scan_lines);

% (2) Frequency filtering (fundamental & 2nd harmonic)
fs  = 1 / kgrid.dt;
scan_lines_fund = gaussianFilter(scan_lines, fs, fc,       100, true);
scan_lines_harm = gaussianFilter(scan_lines, fs, 2 * fc,    30, true);

% (3) Envelope detection (Hilbert)
scan_lines_fund = envelopeDetection(scan_lines_fund);
scan_lines_harm = envelopeDetection(scan_lines_harm);

% (4) Log compression (normalized compression ratio)
compression_ratio = 3;
scan_lines_fund = logCompression(scan_lines_fund, compression_ratio, true);
scan_lines_harm = logCompression(scan_lines_harm, compression_ratio, true);

% (5) Scan conversion (upsample ×2 in scan-line axis)
scale_factor = 2;
scan_lines_fund = interp2(1:kgrid.Nt, (1:number_scan_lines).', ...
    scan_lines_fund, 1:kgrid.Nt, (1:1/scale_factor:number_scan_lines).', 'linear', 0);
scan_lines_harm = interp2(1:kgrid.Nt, (1:number_scan_lines).', ...
    scan_lines_harm, 1:kgrid.Nt, (1:1/scale_factor:number_scan_lines).', 'linear', 0);

%% ------------------------- Display result ------------------------------
depth_mm   = ( (1:kgrid.Nt) * kgrid.dt * c0 / 2 ) * 1e3;        % mm
lateral_mm = ( (1:size(scan_lines_fund,1)) - size(scan_lines_fund,1)/2 ) ...
              * (active_elements * element_width_pts * dx / scale_factor) * 1e3;

figure('Color','w'); 
imagesc(lateral_mm, depth_mm, scan_lines_fund.'); axis image ij; colormap gray;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('B-mode (fundamental)  |  %s', tern(use_gpu_array,'GPU','CPU')));

% optional: overlay inclusion circles (approximate locations)
hold on; th = linspace(0,2*pi,256);
cc = [0, 12; -4, 18; 5, 22]; rr = [3; 2.5; 1.8];
for i = 1:numel(rr)
    plot(cc(i,1) + rr(i)*cos(th), cc(i,2) + rr(i)*sin(th), 'r','LineWidth',1);
end
hold off;

figure('Color','w'); 
imagesc(lateral_mm, depth_mm, scan_lines_harm.'); axis image ij; colormap gray;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title(sprintf('B-mode (2nd harmonic) |  %s', tern(use_gpu_array,'GPU','CPU')));


function s = tern(c,a,b), if c, s=a; else, s=b; end, end
