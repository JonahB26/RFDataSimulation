%% Load required 

addpath("C:\Users\MattC\OneDrive\Elastosynth\Simulator")
addpath("C:\Users\MattC\OneDrive\Elastosynth\Simulator/FEM Interface Windows/")
addpath("C:\Users\MattC\OneDrive\Elastosynth\Simulator/FIELD II Windows")
addpath("C:\Users\MattC\OneDrive\Elastosynth\Simulator\Transducers")
close all
clear all
clc

sim_resolution = [256, 256];

%% Load tumour geometry
% load("C:\Users\MattC\OneDrive\Masters\Data\Breast\Tumour Boundaries/P39-W0-S2-T.mat")
load("C:\Users\1bout\Downloads\P39-W0-S2-T.mat");
TumorArea = imresize(double(TumorArea), sim_resolution);
TumorArea = TumorArea > 0.5;

imshow(TumorArea)
axRes = 257;
latRes = 257;

%% Create Masks 

Masks = zeros(sim_resolution(1), sim_resolution(2), 2);
Masks(:,:,1) = TumorArea;
scaled = imresize(Masks(:,:,1),0.1,"nearest");
masks(:,:,1) = imresize(scaled,size(Masks(:,:,1)),'nearest');

skin_mask = zeros(256,256);
skin_mask(1:10,:) = 1;

Masks(:,:,2) = skin_mask;
Masks = logical(Masks);
figure
imshow(sum(Masks,3),[])

%% Add my own data
clear
clc
%load("C:\Users\1bout\Downloads\benign (1)_preCooper_Coopersadded_Output (1).mat");
load("C:\Users\1bout\Downloads\ClinicalBoundaryConditionsAfterPCA.mat")
load("benign (39)_preCooper_075_thickness_075STIFFNESS_Output 1.mat")
boundarySet = displacementDataSet;
YM_final = Output.YM_Image;
tumorMask = Output.cooperImageData.TumorMask;
ligaments = Output.cooperImageData.cooperMask;
YM_final(ligaments) = 0.01*YM_final(ligaments);
%YM_final = 3000*ones(size(Output.YM_Image));
%YM_final(tumorMask) = 15000;

% YM_final = 20000*ones(size(Output.YM_Image));
% YM_final(background) = 2000;
% YM_final(ligaments) = 2000;
% YM_final(ligaments) = 0.75*3000000;
%YM_final = YM_final(1:350,1:350);
%YM_final = YM_final(1:300,1:300);
figure
imshow(YM_final,[0 20000]);
sim_resolution = [size(YM_final,1) size(YM_final,2)];    
axRes = sim_resolution(1) + 1;
latRes = sim_resolution(2) + 1;
%% Generate  YM

close all

YM = 2000*ones(sim_resolution);

YM(Masks(:,:,1)) = 15000;%10000;
% %YM(Masks(:,:,2)) = 42000;
cooperYM = 0.75*3000000;
%YM(:,19:20) = cooperYM;
%YM(:,56:58) = cooperYM  ;
%YM(:,101:103) = cooperYM;
YM(:,130:132) = cooperYM;
%YM(:,150:152) = cooperYM;
%YM(:,180:182) = cooperYM;
YM(:,200:202) = cooperYM;
YM(19:20,:) = cooperYM;
YM(130:132,:) = cooperYM;





imshow(YM,[0,20000])

YM_noisy = YM + 2000*randn(size(YM));

figure
imshow(YM_noisy,[0,20000])
colorbar
YM_hetero = AddPhantomHeterogeneity(YM_noisy, Masks, [0; 3000; 0], 1100);
YM_final = imgaussfilt(YM_hetero);

figure
imshow(YM_final,[0,20000])
colorbar

%% Run Finite Element Analysis
addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\CoopersLigamentsScripts")
% load("C:\Users\1bout\Downloads\ClinicalBoundaryConditionsAfterPCA.mat")
% boundarySet = displacementDataSet .* -100;
boundary_conditions = BoundaryConditions();
% 
% % boundary_conditions.top_axial = -0.25*ones(1, sim_resolution(1) + 1);   
% % boundary_conditions.bottom_axial = zeros(1, sim_resolution(1) + 1);
% % 
% % boundary_conditions.top_lateral = zeros(1, sim_resolution(2) + 1);   
% % boundary_conditions.bottom_lateral = zeros(1, sim_resolution(2) + 1);
% 
% axial_displacements = repmat(linspace(-0.3,0,axRes)',1,latRes);
% % %lateral_displacements = zeros(axRes,latRes);
% % %lateral_displacements = repmat(linspace(-0.05,0,axRes),1,latRes);
% lateral_displacements = repmat(linspace(-0.05,0,latRes),axRes,1);
% boundary_conditions.top_axial = axial_displacements(1,:);   
% boundary_conditions.bottom_axial = axial_displacements(end,:);
% tas
% boundary_conditions.top_lateral = lateral_displacements(1,:);   
% boundary_conditions.bottom_lateral = lateral_displacements(end,:);
% 
% boundary_conditions.left_axial = axial_displacements(:,1)';
% boundary_conditions.right_axial = axial_displacements(:,end)';
% 
% boundary_conditions.left_lateral = lateral_displacements(:,1)';
% boundary_conditions.right_lateral = lateral_displacements(:,end)';

% boundary_conditions = BoundaryConditions();
% axial_displacements = repmat(linspace(-0.3,0,axRes)',1,latRes);
% lateral_displacements = repmat(linspace(-0.05,0,latRes),axRes,1);
% top_left_corner_axial = 0.22 + (0.25-0.22)*rand(1);
% top_right_corner_axial = 0.19 + (0.23-0.19)*rand(1);
% bottom_left_corner_axial = 0.03 + (0.07-0.03)*rand(1);
% bottom_right_corner_axial = 0.02 + (0.06-0.02)*rand(1);
% top_left_corner_lateral = -(0.07 + (0.09 - 0.07)*rand(1));
% top_right_corner_lateral = 0.09 + (0.11 - 0.09)*rand(1);
% boundary_conditions.left_axial = linspace(-top_left_corner_axial,-bottom_left_corner_axial,axRes);
% boundary_conditions.right_axial = linspace(-top_right_corner_axial,-bottom_right_corner_axial,axRes);
% % %
% boundary_conditions.top_axial = linspace(-top_left_corner_axial,-top_right_corner_axial,latRes);
% boundary_conditions.bottom_axial = linspace(-bottom_left_corner_axial,-bottom_right_corner_axial,latRes);
% %
% boundary_conditions.top_lateral = linspace(top_left_corner_lateral,top_right_corner_lateral,latRes);
% boundary_conditions.bottom_lateral = linspace(top_left_corner_lateral,top_right_corner_lateral,latRes);
%
% boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
% boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
% % %
boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);


analysis_options = FEMOpts("cartesian", sim_resolution(1)+1, sim_resolution(2)+1, "PLANE_STRESS");
material = Material(YM_final, 0.48);

simresult = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,true);

%%

load("L12-3V.mat")

% Calculate Image Options
imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
40/1000, 40/1000,10/1000, 10e4,100);
imageopts.decimation_factor = 2;
imageopts.axial_FOV = 60/1000;
imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
imageopts.slice_thickness = 10/1000;

field_init();

D = imageopts.axial_FOV;
L = imageopts.lateral_FOV;
Z = imageopts.slice_thickness;

[X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
I = ones(220,200);

[phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers);

phantom = Phantom(phantom_positions, phantom_amplitudes);

dispx = interp2(X,Y,simresult.axial_disp,phantom_positions(:,1),phantom_positions(:,3));
dispy = interp2(X,Y,simresult.lateral_disp,phantom_positions(:,1),phantom_positions(:,3));

displacements = zeros(imageopts.n_scatterers, 3);
displacements(:,3) = dispx/1000;
displacements(:,1) = dispy/1000;
displacements(:,2) = 0;

%%

[Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
%% Calculate AM2D

addpath("C:\Users\MattC\OneDrive\Masters\Jonah\Displacement_SRC")
addpath("C:\Users\1bout\Documents\School Documents\SamaniLab\ThirdYearThesis\Displacement_SRC")

Frame1 = Frame1./max(Frame1(:));
Frame2 = Frame2./max(Frame2(:));

Frame2 = Frame2(1:size(Frame1,1),:);

Frame1 = imresize(Frame1, [2500, 256]);
Frame2 = imresize(Frame2, [2500, 256]);

params.probe.a_t = 0.63; % frequency dependent attenuation coefficient, in dB/cm/MHz
params.probe.fc = 5e6; %ultrasound center freq. in MHz
params.probe.fs = 60; % sampling freq. in MHz
params.D = 50;
params.L = 40; 


AM2D = RunAM2D(Frame1, Frame2, params);

AM2D.Axial = AM2D.Axial(43:end-53, 11:end-11);
AM2D.Lateral = AM2D.Lateral(43:end-53, 11:end-11);

%% Reconstruction

addpath("C:\Users\MattC\OneDrive\Masters\Reconstruction\ElastosynthBased")

AM2D.Axial = imresize(AM2D.Axial, [axRes, latRes]);
AM2D.Lateral = imresize(AM2D.Lateral, [axRes, latRes]);

strainA = imgaussfilt(conv2(AM2D.Axial, [-1 0; 1 0],'valid'),2);
strainL = imgaussfilt(conv2(AM2D.Lateral, [-1 1; 0 0],'valid'),2);

figure
subplot(1,2,1)
imshow(strainA,[])
title('Axial Strain')
colorbar

subplot(1,2,2)
imshow(strainL, [])
title('Lateral Strain')
colorbar
%% Repeat with STREAL
addpath("C:\Users\1bout\Documents\School Documents\SamaniLab\ThirdYearThesis\STREAL_SRC\")
%AM2D_disps = AM2D;
%AM2D
[Disp_ax,Disp_lat,strainA,strainL,strainS]...
= prepdispsSTREAL(AM2D.Axial(41:end-60,11:end-10),...
AM2D.Lateral(41:end-60,11:end-10));
% [Disp_ax,Disp_lat,strainA,strainL,strainS]...
% = prepdispsSTREAL(AM2D_disps.Axial,...
% AM2D_disps.Lateral);
figure
subplot(1,2,2)
imshow(Disp_ax,[])
title("Axial Displacement")
colorbar
% subplot(2,2,2)
% imshow(Disp_lat,[])
% title("Lateral Displacement")
% colorbar
subplot(1,2,1)
imshow(strainA,[])
title("Axial Strain")
colorbar
% subplot(2,2,4)
% imshow(strainL,[])
% title("Lateral Strain")
% colorbar

%%

options = ReconOpts(0.01, false, true, "combined", 10, 5);

boundary_conditions = BoundaryConditions();

boundary_conditions.top_axial = AM2D.Axial(1,:);   
boundary_conditions.bottom_axial = AM2D.Axial(end,:);   
boundary_conditions.top_lateral = AM2D.Lateral(1,:);  
boundary_conditions.bottom_lateral = AM2D.Lateral(end,:);  

boundary_conditions.left_axial = AM2D.Axial(:,1)'; 
boundary_conditions.right_axial = AM2D.Axial(:,end)'; 
boundary_conditions.left_lateral = AM2D.Lateral(:,1)'; 
boundary_conditions.right_lateral = AM2D.Lateral(:,end)'; 

YM_Image = 0.5*ones(256,256);

Reconresult = RunReconstruction(options,boundary_conditions,YM_Image,strainA,strainL);
