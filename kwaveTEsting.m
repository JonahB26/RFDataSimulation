addpath('/home/deeplearningtower/MATLAB Add-Ons/Collections/k-wave/K-wave');   % adjust path
savepath   % saves permanently
%%
function [c_map, rho_map] = make_cyst_medium(Nx, Ny, dx, dy, c_bg, rho_bg, opts)
% Create background with weak speckle and an anechoic cyst (c-speed tweak).
% opts: struct with fields:
%   .cyst_center = [x_mm, z_mm]
%   .cyst_radius_mm
%   .dc_inside (e.g., -0.03)   % relative delta c inside cyst
%   .sigma_c (e.g., 0.003)     % small speckle in c
%   .sigma_rho (e.g., 0.01)    % small speckle in rho

if nargin<7, opts = struct; end
get = @(f, d) (isfield(opts,f) * opts.(f) + (~isfield(opts,f))*d);

cx_mm   = get('cyst_center', [0, 12]);   % [x,z] in mm (0=mid)
R_mm    = get('cyst_radius_mm', 3.0);
dc_in   = get('dc_inside', -0.03);       % -3% sound speed inside (soft cyst)
sig_c   = get('sigma_c', 0.003);
sig_rho = get('sigma_rho', 0.01);

x = ((1:Nx) - (Nx+1)/2)*dx;              % meters (lateral)
z = (1:Ny)*dy;                            % meters (depth)
[X,Z] = meshgrid(x,z);                    % Ny×Nx

% base maps with weak speckle
c_map   = c_bg*(1 + sig_c*randn(Ny,Nx));
rho_map = rho_bg*(1 + sig_rho*randn(Ny,Nx));

% cyst mask (anechoic: reduce speckle; add small c-contrast)
xc = cx_mm(1)*1e-3; zc = cx_mm(2)*1e-3; R = R_mm*1e-3;
cyst = ( (X - xc).^2 + (Z - zc).^2 ) <= R^2;

c_map(cyst)   = c_bg*(1 + dc_in);       % slight speed change (edge reflection)
rho_map(cyst) = rho_bg;                  % homogeneous; remove speckle inside
end
%%
%% Grid
Nx=256; Ny=256; dx=50e-6; dy=50e-6;
kgrid = kWaveGrid(Nx,dx,Ny,dy);

%% Medium
c0=1500; rho0=1000;
medium.sound_speed = c0;
medium.density     = rho0;
medium.alpha_coeff = 0.0; medium.alpha_power = 1.5;

%% Linear array (2D)
nElem = 64;
pitch = 0.30e-3;        % lateral center-to-center spacing (m)
width = 0.27e-3;        % element lateral size along x (m)
Lx = width;             % size along x (m)
Ly = dy;                % ~one grid cell along y (m)
z0 = 1.5e-3;            % axial y-position (m), must be within [0, Ny*dy]

% Lateral centers IN-GRID (centered laterally at mid-grid)
x_mid = (Nx*dx)/2;                                      % ~6.4 mm for 256*50µm
x_centers = x_mid + ((-(nElem-1)/2):(+(nElem-1)/2)) * pitch;

% Safety: ensure all are inside the grid bounds
assert(all(x_centers>=0 & x_centers<=Nx*dx), 'Array extends outside grid');
assert(z0>=0 && z0<=Ny*dy, 'z0 outside grid');

arr = kWaveArray();
for e = 1:nElem
    arr.addRectElement([x_centers(e), z0], Lx, Ly, 0);  % (position, Lx, Ly, theta)
end

% PLOT (note the second boolean arg)
arr.plotArray(kgrid, true); axis image ij;
title('Linear array (kWaveArray)'); xlabel('x (m)'); ylabel('y (m)');





%%
% We'll form 256 lines across the sector; you can also do lateral positions.
n_lines = 256;
thetas  = linspace(-10, 10, n_lines)*pi/180;   % ±10° sector

% We'll standardize to 2500 samples per line (resampled after sim)
n_samples_out = 2500;

RF_256x2500 = zeros(n_lines, n_samples_out, 'single');   % your ML input

%%
use_cuda = true;   % set false to use gpuArray path below

if use_cuda
  for i = 1:n_lines
    delays    = arr.getSteeringDelays(kgrid, medium, 'angle', thetas(i));
    source.p  = arr.getDistributedSourceSignal(kgrid, tb, delays);

    inFile  = sprintf('in_%03d.h5',i);
    outFile = sprintf('out_%03d.h5',i);

    % write input, exit without MATLAB solve
    args = {'DataCast','single','PMLInside',false,'PlotPML',false, ...
            'SaveToDisk',true,'SaveToDiskExit',true,'SaveToDiskFilename',inFile};
    kspaceFirstOrder2D(kgrid, medium, source, sensor, args{:});

    % run C++/CUDA solver (ensure it's on your PATH)
    system(sprintf('kspaceFirstOrder-CUDA -i %s -o %s', inFile, outFile));

    % read Nt×Nelem per-element RF for this shot
    P = h5read(outFile, '/p');

    % store for USTB (we'll build P_all after loop to save memory)
    if i==1
      Nt = size(P,1); Ne = size(P,2);
      P_all = zeros(Nt, Ne, n_lines, 'single');
    end
    P_all(:,:,i) = single(P);
  end
else
  % (B) gpuArray (MATLAB) path — slower than CUDA binary for heavy runs
  for i = 1:n_lines
    delays    = arr.getSteeringDelays(kgrid, medium, 'angle', thetas(i));
    source.p  = arr.getDistributedSourceSignal(kgrid, tb, delays);

    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
                    'DataCast','gpuArray-single', 'PMLInside',false, 'PlotPML',false);

    P = gather(sensor_data.p);          % Nt×Nelem
    if i==1
      Nt = size(P,1); Ne = size(P,2);
      P_all = zeros(Nt, Ne, n_lines, 'single');
    end
    P_all(:,:,i) = single(P);
  end
end

dt = kgrid.dt;
fs = 1/dt;
%%
% --- Build USTB channel_data from k-Wave outputs ---
% P_all: Nt × Ne × Ntx
chd = uff.channel_data();
chd.sampling_frequency = fs;
chd.sound_speed        = c_bg;          % average; USTB also supports maps in newer APIs
chd.initial_time       = 0;
chd.data               = P_all;

% Probe descriptor
probe = uff.probe();
probe.N              = Ne;
probe.pitch          = pitch;
probe.element_width  = width;
probe.kerf           = kerf;
chd.probe = probe;

% Transmission sequence (one steered wave per line)
seq(n_lines) = uff.wave();    % preallocate
for k=1:n_lines
    seq(k) = uff.wave();
    seq(k).probe  = probe;
    % Plane-like steered wave; for focal beams you can set .focus
    seq(k).source = uff.source('azimuth', thetas(k), 'distance', Inf);
end
chd.sequence = seq;

% Define a linear scan grid for USTB imaging (display, not your RF matrix)
img_width_m = Ne*pitch;       % simple choice: aperture width
x_axis = linspace(-img_width_m/2, img_width_m/2, 256);
z_max  = 0.04;                % 40 mm display depth (adjust)
z_axis = linspace(0, z_max, 512);
scan = uff.linear_scan('x_axis', x_axis, 'z_axis', z_axis);

% --- DAS beamforming in USTB ---
das = midprocess.das();
das.dimension     = uff.dimension.both;     % TX & RX
das.scan          = scan;
das.channel_data  = chd;

das.receive_apodization.window   = uff.window.hanning;
das.receive_apodization.f_number = 1.5;
das.transmit_apodization.window   = uff.window.hanning;
das.transmit_apodization.f_number = 1.5;

b_data = das.go();  % Complex beamformed data on the scan grid

% --- Visualize a nice B-mode (cyst should appear anechoic with edge) ---
env = abs(hilbert(b_data.data));
env = env / (max(env(:)) + eps);
bmode_db = 20*log10(env + 1e-6);

figure; imagesc(x_axis*1e3, z_axis*1e3, bmode_db);
colormap gray; caxis([-60 0]); axis image ij;
xlabel('Lateral (mm)'); ylabel('Depth (mm)');
title('USTB B-mode (k-Wave data) — cyst phantom');
%%
% We'll take RF A-lines at the 256 x-positions that match the display grid.
RF_lines = zeros(n_lines, n_samples_out, 'single');

% Generate an axial 1D scan object at each x (USTB will beamform there)
for i = 1:n_lines
    x0 = x_axis(i);
    scan_line = uff.linear_scan('x_axis', x0, 'z_axis', linspace(0, z_max, n_samples_out));
    das1 = midprocess.das();
    das1.dimension     = uff.dimension.both;
    das1.scan          = scan_line;
    das1.channel_data  = chd;
    das1.receive_apodization.window   = uff.window.hanning;
    das1.receive_apodization.f_number = 1.5;
    das1.transmit_apodization.window   = uff.window.hanning;
    das1.transmit_apodization.f_number = 1.5;

    rf_b = das1.go();                 % complex samples vs depth
    RF_lines(i,:) = single(real(rf_b.data(:)).');   % raw RF (pre-envelope)
end

% If you want exactly 2500 samples (already set above), RF_lines is 256×2500
RF_256x2500 = RF_lines;   % this is your ML-ready RF frame
