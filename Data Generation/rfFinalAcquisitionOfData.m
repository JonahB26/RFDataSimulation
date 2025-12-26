% Script is only dependant on the loaded data in line 7. For-loop will be
% easy. For assistance, refer to ManualClinicalSimulation in Elastosynth.
clear
clc

dataResultsFolder = 'C:/Users/jboutin2/Documents/ISBI_Results/';
% 
% Here we'll create a folder and directory for the files
% DataFolder = 'C:/Users/jboutin2/Documents/JonahCode/OutputYMs';
DataFolder = 'C:/Users/jboutin2/Documents/TestFilesMat/OutputTests';
Files = dir(fullfile(DataFolder,'*.mat'));
for i = 6%:3%:1%length(Files)
    tic
    clc
    
    %Uncomment later, was just for test.
    currentFile = Files(i).name
    [~,currentFileName,ext] = fileparts(currentFile);
    disp(currentFileName)

    %
    % Here we'll load the corresponding file, Output.YM_Image is the input for
    % material.
    
    
    % load(fullfile(uigetdir,uigetfile));
    %load('YM_HalfThick.mat');
    %uncomment later
    load(fullfile(DataFolder,currentFile));

   % load('YM_testHalf.mat');

    %load("C:\Users\1bout\Downloads\benign (1)_preCooper_Coopersadded_Output (1).mat");%First one for a test.
    
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\CoopersLigamentsScripts")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\CoopersLigamentsScripts\FIELD II Windows")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\Elastosynth\Simulator\")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\Elastosynth\Simulator\FEM Interface Windows")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\Elastosynth\Simulator\FEM_src/")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\Elastosynth\Simulator\FIELD II Windows")
    % addpath("C:\Users\1bout\Documents\School Documents\SamaniLab\ThirdYearThesis\Displacement_SRC")
    % addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\CoopersLigamentsScripts")
    
    transducer_list = genTransducers(); % Run the file to generate transducer list.
    
    % % Just a test, modify to binary image.
    % YM_final = zeros(size(Output.YM_Image));
    % YM_final(:) = 1000;
    % YM_final(Output.Image.cooperImageData.cooperMask) = 1000;
    % YM_final(Output.Image.TumorMask) = 5000;

    %uncomment later
   %YM_final = Output.YM_Image;
   %YM_final(Output.Image.cooperImageData.cooperMask) = 3000;
   YM_final = 3000*ones(471,562);
   YM_final(Output.Image.cooperImageData.cooperMask) = 3000;
   YM_final(Output.Image.TumorMask) = 15000;

   %YM_final = YM_Image;
    sim_resolution = [size(YM_final,1) size(YM_final,2)]; 
    
    axRes = size(YM_final,1) + 1;
    latRes = size(YM_final,2) + 1;
    
    boundary_conditions = BoundaryConditions();
    
    axial_displacements = repmat(linspace(-0.3,0,axRes)',1,latRes);
    lateral_displacements = repmat(linspace(-0.05,0,latRes),axRes,1);

    top_left_corner_axial = 0.22 + (0.25-0.22)*rand(1);
    top_right_corner_axial = 0.19 + (0.23-0.19)*rand(1);

    bottom_left_corner_axial = 0.03 + (0.07-0.03)*rand(1);
    bottom_right_corner_axial = 0.02 + (0.06-0.02)*rand(1);

    top_left_corner_lateral = -(0.07 + (0.09 - 0.07)*rand(1));
    top_right_corner_lateral = 0.09 + (0.11 - 0.09)*rand(1);


  %   boundary_conditions.left_axial = linspace(-top_left_corner_axial,-bottom_left_corner_axial,axRes);
  %   boundary_conditions.right_axial = linspace(-top_right_corner_axial,-bottom_right_corner_axial,axRes);
  % %   % 
  % boundary_conditions.top_axial = linspace(-top_left_corner_axial,-top_right_corner_axial,latRes);
  %   boundary_conditions.bottom_axial = linspace(-bottom_left_corner_axial,-bottom_right_corner_axial,latRes);
  % % 
  %   boundary_conditions.top_lateral = linspace(top_left_corner_lateral,top_right_corner_lateral,latRes);
  %   boundary_conditions.bottom_lateral = linspace(top_left_corner_lateral,top_right_corner_lateral,latRes);
  % 
  %   boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
  %   boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
  %   % % 
    boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);


    boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);

   boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
    boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);

    boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
    boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);

    boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
    boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);

   % 

    % Change this so that i also have bottom left/right
    % after, need to run it, also see without ligaments, if different
    % need to try with 0.5 thickness then with 0.75 of the Cooper 
    % YM and then with both these options, send to SAMANI
    % 

    
   % boundary_conditions.top_axial = axial_displacements(1,:); 
   % 
   %  boundary_conditions.bottom_axial = axial_displacements(end,:);
   % 
   %  boundary_conditions.top_lateral = lateral_displacements(1,:);   
   %  boundary_conditions.bottom_lateral = lateral_displacements(end,:);
   % 
   %  boundary_conditions.left_axial = axial_displacements(:,1)';
   %  boundary_conditions.right_axial = axial_displacements(:,end)';
   % 
   %  boundary_conditions.left_lateral = lateral_displacements(:,1)';
   %  boundary_conditions.right_lateral = lateral_displacements(:,end)';
    
    analysis_options = FEMOpts("cartesian", axRes, latRes, "PLANE_STRESS");
    material = Material(YM_final, 0.48);
    
    disp("Running FEA...")
    simresult = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,true);
    disp("FEA complete.")
 %%   
    randomTransducerNum = randi([1 3],1);
    transducer = transducer_list{randomTransducerNum};
    
    % Calculate Image Options
    imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
    40/1000, 40/1000,10/1000, 10e4,100);
    imageopts.decimation_factor = 2;
    imageopts.axial_FOV = 60/1000;
    imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
    imageopts.slice_thickness = 10/1000;
    
    field_init();
    
    D = 60/1000;
    L = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
    Z = 10/1000;
    
    [X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
    I = ones(220,200);
    % load('testhalfCooperData.mat'); % comment later.
    % tumorMask = imread('benign (8)_mask.png'); %comment later
    % tumorMask = logical(tumorMask);
    % I(Output.cooperImageData.cooperMask) = 1; % NEED TO CHECK THESE LINES
    % I(Output.cooperImageData.TumorMask) = 50;
    % I(Output.Image.Coopermask) = 7;
    % I(Output.Image.TumorMask) = 5;
   % I(cooperImageData.cooperMask) = 7;
    %I(tumorMask) = 5;
    
    [phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers);
    
    phantom = Phantom(phantom_positions, phantom_amplitudes);
    
    dispx = interp2(X,Y,simresult.axial_disp,phantom_positions(:,1),phantom_positions(:,3));
    dispy = interp2(X,Y,simresult.lateral_disp,phantom_positions(:,1),phantom_positions(:,3));
    
    displacements = zeros(imageopts.n_scatterers, 3);
    displacements(:,3) = dispx/1000;
    displacements(:,1) = dispy/1000;
    displacements(:,2) = 0;
    
    figure
    % % scatter(phantom_positions(:,3)-30/1000, phantom_positions(:,1), 8, displacements(:,3),'filled','o')
    % scatter( phantom_positions(:,1),phantom_positions(:,3)-30/1000, 8, displacements(:,3),'filled','o')
    subplot(1,2,1)
    scatter(phantom_positions(:,1),phantom_positions(:,3),3,dispx)
    title('Scatterer Axial Displacements','FontSize',20)
    colorbar

    subplot(1,2,2)
    scatter(phantom_positions(:,1),phantom_positions(:,3),3,dispy)
    title('Scatterer Lateral Displacements','FontSize',20)
    colorbar

    disp("Generating RF data.")
    [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
    disp("RF generation complete.")
    
    Frame1 = Frame1./max(Frame1(:));
    Frame2 = Frame2./max(Frame2(:));
    
    if size(Frame1,1) > size(Frame2,1)
        Frame1 = Frame1(1:size(Frame2,1),:);
    else
        Frame2 = Frame2(1:size(Frame1,1),:);
    end    
    Frame1 = imresize(Frame1, [2500, 256]);
    Frame2 = imresize(Frame2, [2500, 256]);
    
    %% Visualization
    Frame1 = Frame1./max(Frame1(:));
    Frame2 = Frame2./max(Frame2(:));
    
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
    
    %% Reconstruction, comment out visualization for mass gen
        
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

    %% Reformat and Visualize the data

% %AM2D
% AM2D_disps = AM2D;
% [Disp_ax,Disp_lat,strainA,strainL,strainS]...
%               = prepdispsforreconstruction(AM2D_disps.Axial(41:end-60,11:end-10),...
%               AM2D_disps.Lateral(41:end-60,11:end-10));
% 
% figure
% subplot(2,2,1)
% imshow(Disp_ax,[])
% title("Axial Displacement")
% colorbar
% 
% subplot(2,2,2)
% imshow(Disp_lat,[])
% title("Lateral Displacement")
% colorbar
% 
% subplot(2,2,3)
% imshow(strainA,[])
% title("Axial Strain")
% colorbar
% 
% subplot(2,2,4)
% imshow(strainL,[])
% title("Lateral Strain")
% colorbar

%% Repeat with STREAL
 %AM2D_disps = AM2D;
%AM2D
[Disp_ax,Disp_lat,strainA,strainL,strainS]...
              = prepdispsSTREAL(AM2D.Axial(41:end-60,11:end-10),...
              AM2D.Lateral(41:end-60,11:end-10));

% [Disp_ax,Disp_lat,strainA,strainL,strainS]...
%               = prepdispsSTREAL(AM2D_disps.Axial,...
%               AM2D_disps.Lateral);

figure
subplot(2,2,1)
imshow(Disp_ax,[])
title("Axial Displacement")
colorbar

subplot(2,2,2)
imshow(Disp_lat,[])
title("Lateral Displacement")
colorbar

subplot(2,2,3)
imshow(strainA,[])
title("Axial Strain")
colorbar

subplot(2,2,4)
imshow(strainL,[])
title("Lateral Strain")
colorbar

    %% Save the data

    FEA_FIELDII_Results = struct();
    
    FEA_FIELDII_Results.axnodalresolution = axRes;
    FEA_FIELDII_Results.latnodalresolution = latRes;

    FEA_FIELDII_Results.InitialBoundaryConditions.top_axial = boundary_conditions.top_axial;
    FEA_FIELDII_Results.InitialBoundaryConditions.bottom_axial = boundary_conditions.bottom_axial;
    FEA_FIELDII_Results.InitialBoundaryConditions.left_axial = boundary_conditions.left_axial;
    FEA_FIELDII_Results.InitialBoundaryConditions.right_axial = boundary_conditions.right_axial;

    FEA_FIELDII_Results.InitialBoundaryConditions.top_lateral = boundary_conditions.top_lateral;
    FEA_FIELDII_Results.InitialBoundaryConditions.bottom_lateral = boundary_conditions.bottom_lateral;
    FEA_FIELDII_Results.InitialBoundaryConditions.left_lateral = boundary_conditions.left_lateral;
    FEA_FIELDII_Results.InitialBoundaryConditions.right_lateral = boundary_conditions.right_lateral;

    FEA_FIELDII_Results.FEMOutput.axial_disp = simresult.axial_disp;
    FEA_FIELDII_Results.FEMOutput.lateral_disp = simresult.lateral_disp;
    FEA_FIELDII_Results.FEMOutput.axial_strain = simresult.axial_strain;
    FEA_FIELDII_Results.FEMOutput.lateral_strain = simresult.lateral_strain;
    FEA_FIELDII_Results.FEMOutput.shear_strain = simresult.shear_strain;
    FEA_FIELDII_Results.FEMOutput.axial_stress = simresult.axial_stress;
    FEA_FIELDII_Results.FEMOutput.lateral_stress = simresult.lateral_stress;
    
    FEA_FIELDII_Results.OriginalFileName = strcat(currentFileName,ext);
    
    FEA_FIELDII_Results.Imageinfo.YM_Image = Output.YM_Image;
    FEA_FIELDII_Results.Imageinfo.minTumorYM = Output.minTumorYM;
    FEA_FIELDII_Results.Imageinfo.maxTumorYM = Output.maxTumorYM;
    FEA_FIELDII_Results.Imageinfo.uint8img = Output.cooperImageData.originalImg;
    FEA_FIELDII_Results.Imageinfo.TumorMask = Output.cooperImageData.TumorMask;
    FEA_FIELDII_Results.Imageinfo.CooperMask = Output.cooperImageData.cooperMask;

    FEA_FIELDII_Results.FIELDIIinfo.Frame1 = Frame1;
    FEA_FIELDII_Results.FIELDIIinfo.Frame2 = Frame2;
    FEA_FIELDII_Results.FIELDIIinfo.AM2Dresults.Axialdisp = AM2D.Axial;
    FEA_FIELDII_Results.FIELDIIinfo.AM2Dresults.Lateraldisp = AM2D.Lateral;
    FEA_FIELDII_Results.FIELDIIinfo.AM2Dresults.AxialStrainForVisualization = strainA;
    FEA_FIELDII_Results.FIELDIIinfo.AM2Dresults.LateralStrainForVisualization = strainL;

    FEA_FIELDII_Results.FIELDIIinfo.transducer = transducer;
    FEA_FIELDII_Results.FIELDIIinfo.AM2Dresults.params = params;
    FEA_FIELDII_Results.FIELDIIinfo.Scatterers.positions = phantom.positions;
    FEA_FIELDII_Results.FIELDIIinfo.Scatterers.amplitudes = phantom.amplitudes;
  
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.no_lines = imageopts.no_lines;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.image_width = imageopts.image_width;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.decimation_factor = imageopts.decimation_factor;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.d_x = imageopts.d_x;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.axialFOV = imageopts.axial_FOV;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.lateralFOV = imageopts.lateral_FOV;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.slice_thickness = imageopts.slice_thickness;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.n_scatterers = imageopts.n_scatterers;
    FEA_FIELDII_Results.FIELDIIinfo.imageopts.speed_factor = imageopts.speed_factor;


    file_hex = DataHash(FEA_FIELDII_Results, 'array','hex');
    
    filename = strcat(dataResultsFolder, file_hex,".mat");
    %filename = strcat(uigetdir, file_hex,".mat");
    save(filename, "FEA_FIELDII_Results")
    toc
end