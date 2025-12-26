% This is just a test script to try and acquire RF data.

% You will probably encounter an error in linux if you try to build the library
% because of a conflict between versions of a dependancy, run this command to fix it
%export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 matlab
% First get boundary conditions

% Create displacements and RF folders and provide their paths here.
%dataResultsFolder = 

% Here we'll create a folder and directory for the files
%DataFolder = ;
%Files = dir(fullfile(DataFolder,'*.mat'));
%currentFile = Files(i)
%[~,currentFileName,ext] = fileparts(currentFile)
%
% Here we'll load the corresponding file, Output.YM_Image is the input for
% material.
clear
clc
load(fullfile(uigetdir,uigetfile));
%data = load(Files(i));
%YM_Image = data.Output.YM_Image;
YM_Image = Output.YM_Image; % For test, delete after.

%% Define the nodal resolution and and arbitrarily create displacements for
% BCs
axRes = size(YM_Image,1) + 1;
latRes = size(YM_Image,2) + 1;
axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
lateral_displacements = zeros(axRes,latRes);
%%
% Set material and analysis options for FEM, and then configure BCs
ConfigureFEM();

% Create material definition
material = clib.FEM_Interface.Material_MATLAB;

material.youngs_modulus = flatten(YM_Image);
material.poissons_ratio = flatten(0.48*ones(size(axial_displacements)-1));

% Analysis options
analysis_options = clib.FEM_Interface.AnalysisOptions;
analysis_options.coordinate_system_type = "cartesian";
analysis_options.element_type = "PLANE_STRAIN";
analysis_options.axial_nodal_resolution = axRes;
analysis_options.lateral_nodal_resolution = latRes;

% Boundary Conditions
boundary_conditions = clib.FEM_Interface.BoundaryStruct;
%%
% Set BCs from the displacements
boundary_conditions.top_axial = axial_displacements(1,:);   
boundary_conditions.bottom_axial = axial_displacements(end,:);
boundary_conditions.top_lateral = lateral_displacements(1,:);   
boundary_conditions.bottom_lateral = lateral_displacements(end,:);
% % Analysis options definition
% analysis_options = clib.FEM_Interface.AnalysisOptions;
% analysis_options.coordinate_system_type = "cartesian";
% analysis_options.element_type = "PLANE_STRAIN";
% analysis_options.axial_nodal_resolution = size(YM_Image,1) + 1;
% analysis_options.lateral_nodal_resolution = size(YM_Image,2) + 1;
% 
% % Make the axial/lateral disp size, for use in the FEM function
% axRes = analysis_options.axial_nodal_resolution;
% latRes = analysis_options.lateral_nodal_resolution;
% 
% % Material definition
% material = clib.FEM_Interface.Material_MATLAB;
% material.youngs_modulus = flatten(YM_Image);
% material.poissons_ratio = flatten(0.48*ones(size(YM_Image)));
% 
% % Boundary conditions definition
% boundary_conditions = clib.FEM_Interface.BoundaryStruct;
% 
% axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
% lateral_displacements = zeros(axRes,latRes);
% 
% boundary_conditions.top_axial = axial_displacements(1,:);   
% boundary_conditions.bottom_axial = axial_displacements(end,:);
% boundary_conditions.top_lateral = lateral_displacements(1,:);   
% boundary_conditions.bottom_lateral = lateral_displacements(end,:);

% %Lateral boundary conditions
% boundary_conditions.left_lateral = displacementDataSet(i,1:200);
% 
% boundary_conditions.right_lateral = displacementDataSet(i,201:400);
% 
% boundary_conditions.bottom_lateral = displacementDataSet(i,401:600);
% 
% boundary_conditions.top_lateral = displacementDataSet(i,601:800);
% 
% %Interpolate the boundary conditions so they match the size of YM_Image.
% boundary_conditions.left_lateral = InterpAndAddPoints(boundary_conditions.left_lateral, ...
%     analysis_options.axial_nodal_resolution);
% boundary_conditions.right_lateral = InterpAndAddPoints(boundary_conditions.right_lateral, ...
%     analysis_options.axial_nodal_resolution);
% boundary_conditions.bottom_lateral = InterpAndAddPoints(boundary_conditions.bottom_lateral, ...
%     analysis_options.lateral_nodal_resolution);
% boundary_conditions.top_lateral = InterpAndAddPoints(boundary_conditions.top_lateral, ...
%     analysis_options.lateral_nodal_resolution);
% 
% %Axial boundary conditions
% boundary_conditions.left_axial = displacementDataSet(i,801:1000);
% 
% boundary_conditions.right_axial = displacementDataSet(i,1001:1200);
% 
% boundary_conditions.bottom_axial = displacementDataSet(i,1201:1400);
% 
% boundary_conditions.top_axial = displacementDataSet(i,1401:1600);
% 
% %Interpolate the boundary conditions so they match the size of YM_Image.
% boundary_conditions.left_axial = InterpAndAddPoints(boundary_conditions.left_axial, ...
%     analysis_options.axial_nodal_resolution);
% boundary_conditions.right_axial = InterpAndAddPoints(boundary_conditions.right_axial, ...
%     analysis_options.axial_nodal_resolution);
% boundary_conditions.bottom_axial = InterpAndAddPoints(boundary_conditions.bottom_axial, ...
%     analysis_options.lateral_nodal_resolution);
% boundary_conditions.top_axial = InterpAndAddPoints(boundary_conditions.top_axial, ...
%     analysis_options.lateral_nodal_resolution);
%%
% Run FEA and save
tic
clc
disp('Running simulation...')
output = clib.FEM_Interface.RunFiniteElementAnalysis(boundary_conditions,...
                                                      size(axial_displacements),...
                                                      material, analysis_options);

% Comment all this if not visualizing.
axial_disp = reshape(output.axial_displacements.double(), axRes, latRes);
lateral_disp = reshape(output.lateral_displacements.double(), axRes, latRes);

axial_strain = reshape(output.axial_strain.double(),axRes-1, latRes-1);
lateral_strain = reshape(output.lateral_strain.double(),axRes-1, latRes-1);
shear_strain = reshape(output.shear_strain.double(),axRes-1, latRes-1);

axial_stress = reshape(output.axial_stress.double(),axRes-1, latRes-1);
lateral_stress = reshape(output.lateral_stress.double(),axRes-1, latRes-1);

% Comment all this if not visualizing.

    % figure
    % subplot(2,2,1)
    % imshow(axial_disp,[])
    % colorbar
    % title("Axial Displacement", "FontSize",20)
    % 
    % subplot(2,2,2)
    % imshow(lateral_disp,[])
    % colorbar
    % title("Lateral Displacement", "FontSize",20)
    % 
    % subplot(2,2,3)
    % imshow(axial_strain,[])
    % colorbar
    % title("Axial Strain", "FontSize",20)
    % 
    % subplot(2,2,4)
    % imshow(lateral_strain,[])
    % colorbar
    % title("Lateral Strain", "FontSize",20)

FEAResults = struct();

FEAResults.axnodalresolution = axRes;
FEAResults.latnodalresolution = latRes;

FEAResults.FEMOutput.axial_disp = result.axial_disp;
FEAResults.FEMOutput.lateral_disp = result.lateral_disp;
FEAResults.FEMOutput.axial_strain = result.axial_strain;
FEAResults.FEMOutput.lateral_strain = result.lateral_strain;
FEAResults.FEMOutput.shear_strain = result.shear_strain;
FEAResults.FEMOutput.axial_stress = result.axial_stress;
FEAResults.FEMOutput.lateral_stress = result.lateral_stress;
FEAResults.FEMOutput.shear_stress = result.shear_stress;

FEAResults.OriginalFileName = strcat(currentFileName,ext);

FEAResults.Imageinfo = Output;
file_hex = DataHash(output, 'array','hex');
    
filename = strcat(dataResultsFolder, file_hex,".mat");
%filename = strcat(uigetdir, file_hex,".mat");
save(filename, "FEAResults")
%% FIELD II Portion
%load()%Put the path to the previously saved transducer list here.
speed_factor = 100;
field_init()

transducernum = randi([1 3],1);
transducer = transducer_list{1,transducernum};

D = 60/1000;
L = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
Z = 10/1000;

[X,Y] = meshgrid(linspace(L/2,-L/2,latRes),linspace(D,0,axRes)+0.03);

I = double(Output.Image.loaded);

close all

figure
[phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, 10e4);

phantom = Phantom(phantom_positions, phantom_amplitudes);

imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
    40/1000, 40/1000,10/1000, 10e4,100);
imageopts.decimation_factor = 2;
iamgeopts.axial_FOV = 60/1000;
imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
iamgeopts.slice_thickness = 10/1000;

dispx = interp2(X,Y,axial_disp,phantom_positions(:,1),phantom_positions(:,3));
dispy = interp2(X,Y,lateral_disp,phantom_positions(:,1),phantom_positions(:,3));

figure
subplot(1,2,1)
scatter(phantom_positions(:,1),phantom_positions(:,3),3,dispx)
title("Scatterer Axial Displacements", "FontSize", 20)
colorbar

subplot(1,2,2)
scatter(phantom_positions(:,1),phantom_positions(:,3),3,dispy)
title("Scatterer Lateral Displacements", "FontSize", 20)
colorbar

displacements = zeros(10e4, 3);
displacements(:,3) = dispx/1000;
displacements(:,1) = dispy/1000;

tic
[Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, speed_factor);
toc

size(Frame1)
Frame1 = Frame1(1:2900,:);
Frame1 = Frame1 ./ max(Frame1(:));

size(Frame2)
Frame2 = Frame2(1:2900,:);
Frame2 = Frame2 ./ max(Frame2(:));

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

FIELD2Results.FEAResults = FEAResults;

FIELD2Results.transducer = transducer;
FIELD2Results.phantom_positions = phantom_positions;
FIELD2Results.phantom_amplitudes = phantom_amplitudes;
FIELD2Results.dispx = dispx;
FIELD2Results.dispy = dispy;
FIELD2Results.displacements = displacements;

FIELD2Results.Frame1 = Frame1;
FIELD2Results.Frame2 = Frame2;

FIELD2Results.BMODE1.BMODE1 = BMODE1;
% FIELD2Results.BMODE1.variable_for_imshow = imresize(BMODE1, [220,200]);

FIELD2Results.BMODE2.BMODE2 = BMODE2;
% FIELD2Results.BMODE2.variable_for_imshow = imresize(BMODE1, [220,200]);









































