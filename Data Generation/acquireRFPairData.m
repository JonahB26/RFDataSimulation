% This script is responsible for creating the RF frane pair data. It will use
% the phantom function, the ImageOpts class, the Transducer class, the Image
% to scatterers function, and the generate frame pair linear function.

field_init();

addpath(genpath("C:\Users\1bout\OneDrive\SamaniLab\Elastosynth\Simulator"));


% I = Output.Image.loaded;
% I = double(I);
%I = test;
I = ones(585,683);

% axialdisp = Output.FEAdisplacements.AxialDisplacements;
% lateraldisp = Output.FEAdisplacements.LateralDisplacements;
% axialdisp = resultnoCooper.axial_disp;
% lateraldisp = resultnoCooper.lateral_disp;
axialdisp = FEAResults.FEMOutput.axial_disp;
lateraldisp = FEAResults.FEMOutput.lateral_disp;

axialresolution = size(axialdisp,1);
lateralresolution = size(axialdisp,2);

%Add shit to imageopts here
imageopts.no_lines = 256;
imageopts.image_width = 40/1000;
imageopts.decimation_factor = 2;
imageopts.axial_FOV = 40/1000;
imageopts.lateral_FOV = 40/1000;
imageopts.slice_thickness = 10/1000;
imageopts.speed_factor = 100;
imageopts.n_scatterers = 114000;
imageopts.d_x = imageopts.image_width / imageopts.no_lines;

D = imageopts.axial_FOV;
L = imageopts.lateral_FOV;
Z = imageopts.slice_thickness;

%Start the for loop here.  
%Randomly pick the transducer and get the data.
%transducerNum = randi([1 2],1);

transducer = transducer_list{1};

% Update the MetaData.FEM_resolution with the resolution of the current
% image.

%load the image here, use the .mat files.

% [X,Y] = meshgrid(linspace(-L/2,L/2,axialresolution+1),linspace(0,D,lateralresolution+1)+0.03);
[X,Y] = meshgrid(linspace(-L/2,L/2,lateralresolution),linspace(0,D,axialresolution)+0.03);

[phantom_positions, phantom_amplitudes] = ImageToScatterers(I,D,L,Z,imageopts.n_scatterers);

phantom = Phantom(phantom_positions,phantom_amplitudes);

%Update these two lines with the fem data.
dispx = interp2(X,Y,axialdisp,phantom_positions(:,1),phantom_positions(:,3));
dispy = interp2(X,Y,lateraldisp,phantom_positions(:,1),phantom_positions(:,3));

displacements = zeros(imageopts.n_scatterers, 3);
displacements(:,3) = dispx/1000;
displacements(:,1) = dispy/1000;
%Need to figure out what to do with the OOP thing.
% displacements(:,2) = parameter_table.OOP_displacement(i)/1000 * randn(imageopts.n_scatterers,1);
displacements(:,2) = 10/1000 * randn(imageopts.n_scatterers,1);

figure
scatter(phantom_positions(:,3)-30/1000, phantom_positions(:,1), 8, displacements(:,3),'filled','o')
title('Scatterer Displacements')
colorbar

[Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);

figure
subplot(1,2,1)
BMODE1 = log(abs(hilbert(Frame1/max(Frame1(:))))+.01);
imshow(imresize(BMODE1, [220,200]),[])
title('Frame 1')
colorbar

subplot(1,2,2)
BMODE1 = log(abs(hilbert(Frame2/max(Frame2(:))))+.01);
imshow(imresize(BMODE1, [220,200]),[])
title('Frame 2')
colorbar
sgtitle('CooperYM = 2.5 - 3.5MPa')

% Save the file here.

