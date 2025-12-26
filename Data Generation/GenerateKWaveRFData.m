
% Extract Field II transducer parameters
transducer = genTransducers(); % match Field II code
transducer = transducer{1};
% Grid size and spacing
Nx = 256; dx = 1e-4;
Ny = 2500; dy = 1e-4; % Matches resized Field II frame size

kgrid = kWaveGrid(Nx, dx, Ny, dy); % Note: k-Wave Y is axial, X is lateral

% Time array
c0 = transducer.speed_of_sound;
t_end = 2 * Ny * dy / c0;
kgrid.setTime(round(t_end / (1 / transducer.sampling_frequency)), 1 / transducer.sampling_frequency);


% Generate image-based tissue map
load("P39-W0-S2-T.mat");
tumor_mask = TumorArea;
tumor_mask = imresize(double(tumor_mask),[256,256]);
tumor_mask = tumor_mask > 0.5;
lateralres = size(tumor_mask,2); %This line and next obtains resolution
axialres = size(tumor_mask,1);
ligament_thickness = 0.4475;
ligament_stiffness = 150000;

imshow(tumor_mask);
hold on;

[cooper_mask,~] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness,'top-right'); %Add the ligaments
I = zeros(256, 256);
I(tumor_mask) = 1;
I(cooper_mask) = 2;

% Define physical properties
c_map = ones(size(I)) * 1540;
rho_map = ones(size(I)) * 1000;
amp_map = ones(size(I)) * 0;

c_map(I == 1) = 1580; % Tumor
rho_map(I == 1) = 1050;
amp_map(I == 1) = 1.0;

c_map(I == 2) = 1450; % Coopers
rho_map(I == 2) = 1100;
amp_map(I == 2) = 0.8;

% Resize and pad to k-Wave size
c_map = imresize(c_map, [Ny, Nx]);
rho_map = imresize(rho_map, [Ny, Nx]);
amp_map = imresize(amp_map, [Ny, Nx]);

medium.sound_speed = c_map;
medium.density = rho_map;
