% This script is responsible for calculating the mutual information between
% two clinical simulated images, so that I can then use that I value to
% optimize the variables of cooper YM and cooper ligament thickness for my
% own dataset. Created by Jonah Boutin on 11/22/2024.

% Clear the workspace
clear,clc,close all

%% Step 1
% First, insert code to get a 3D array of various clinical boundaries.

% The following resizes the tumor boundaries to 256x256 logical image.
% Let's just run the script on four of them, and average the MI values.
addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\MESc Code Files - git\Data Generation\ClinicalTumorBoundariesForMutualInfoCalc") %Add tumor boundary path
% clinicalFilePath1 = 'P102-W0-S2-T.mat';
% clinicalFilePath2 = 'P107-W1-S2-T.mat';
% clinicalFilePath3 = 'P113-W4-S2-t.mat';
% clinicalFilePath4 = 'P122-W2-S2-T.mat';
% 
% % Create array of the path strings
% clinicalFilePaths = {clinicalFilePath1,clinicalFilePath2,clinicalFilePath3,clinicalFilePath4};

%file_paths = dir(fullfile('C:\Users\jboutin2\Documents\JonahCode\BreastCancerDiagnosis-ML\Data Generation\ClinicalTumorBoundariesForMutualInfoCalc','*.mat'));
file_paths = dir(fullfile('ClinicalTumorBoundariesForMutualInfoCalc/','*.mat')); %For at home
TumorAreas = zeros(256,256,length(file_paths));

% Rip through the files and update TumorAreas so I have array of clinical boundaries
for i = 1%:length(file_paths)
    currentTumorArea = load(file_paths(i).name);
    currentTumorArea = currentTumorArea.TumorArea;
    currentTumorArea = imresize(double(currentTumorArea),[256,256]);
    currentTumorArea = currentTumorArea > 0.5;
    TumorAreas(:,:,i) = currentTumorArea;
end

% Set the YM, this is going to be based on what I'm using in my data.
simulated_YM_matrices = zeros(256,256,size(TumorAreas,3));
mutual_information_values = zeros(length(file_paths),1); % Array to stor mutual information values

for i = 1%:size(simulated_YM_matrices,3)

    current_YM_image = TumorAreas(:,:,i); %This is the image I'm modifing for YM

    randTumorYM = 5000 + (15000 - 5000)*rand(1);%Generate random YM between 5-15kPa

    %Set range for tumor YM throughout image
    minTumorYM = randTumorYM - 1500; 
    maxTumorYM = randTumorYM + 1500;
    
    % Modify the YM image
    current_YM_image(current_YM_image==1) = minTumorYM + (maxTumorYM - minTumorYM)*rand(sum(current_YM_image==1,'all'),1); %Set the tumor YM values, the end is just summing all the 1 values for dimensions sake
    current_YM_image(current_YM_image==0) = 2500 + (3500 - 2500)*rand(sum(current_YM_image==0,"all"),1); %Add in the background YM
    
    % testing
    % current_YM_image(current_YM_image==1) = 15000;
    % current_YM_image(current_YM_image==0) = 2500 + (3500 - 2500)*rand(sum(current_YM_image==0,"all"),1); %Add in the background YM
    % 
    simulated_YM_matrices(:,:,i) = current_YM_image; %Append the overall array

end
disp('Done Step 1.')
%% Step 2
% Then, insert code that calculates FEA and FIELDII on the image.
% NOTE: Always have to go to simulator & displacement and add all folders/subfolders to
% path first!!!!!!!!

% i = 1; %This is just a test, then it will be a for-loop**********************************
for i = 1%:length(file_paths)
    simresults = zeros(1,length(file_paths)); %Will hold the result object for the simulations
    
    % Define the axial/lateral resolution
    axRes = size(simulated_YM_matrices,1) + 1; %Increment by one for calculation
    latRes = size(simulated_YM_matrices,2) + 1;
    sim_resolution = [axRes-1,latRes-1]; %Size of simulation
    
    % Define the boundary conditions, these have alread been decided and are in
    % the ManualClinicalSimulation as well
    boundary_conditions = BoundaryConditions(); %Instance of Boundary Conditions class
    boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
    boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
    boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
    boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
    boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
    boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
    boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
    boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
    
    % Define the analysis options, material, and run FEA
    analysis_options = FEMOpts("cartesian", sim_resolution(1)+1, sim_resolution(2)+1, "PLANE_STRESS");
    material = Material(simulated_YM_matrices(:,:,i), 0.48);
    clc
    disp('Running FEA...')
    fprintf('On number %d\n',i);
    % disp(numIterationText)
    if nnz(mutual_information_values) > 1
        miText = sprintf('Last mutual info value was %d',mutual_information_values(i-1));
        disp(miText)
    end
    simresult = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,true);
    clc
    disp('Done Step 2.')
    %%
    
    % NOW FOR FIELDII
    
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
    
    clc
    disp('Running FIELD II...')
    fprintf('On number %d\n',i);
    % disp(numIterationText)
    if nnz(mutual_information_values) > 1
        miText = sprintf('Last mutual info value was %d',mutual_information_values(i-1));
        disp(miText)
    end
    [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
    
    % Process FIELD II result
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
    
    AM2D.Axial = imresize(AM2D.Axial, [axRes, latRes]);
    AM2D.Lateral = imresize(AM2D.Lateral, [axRes, latRes]);
    
    % strainA = imgaussfilt(conv2(AM2D.Axial, [-1 0; 1 0],'valid'),2);
    % strainL = imgaussfilt(conv2(AM2D.Lateral, [-1 1; 0 0],'valid'),2);
    
    % figure
    % subplot(1,2,1)
    % imshow(strainA,[])
    % title('Axial Strain')
    % colorbar
    % 
    % subplot(1,2,2)
    % imshow(strainL, [])
    % title('Lateral Strain')
    % colorbar
    
    clc
    disp('Done Step 3.')
    
    %%
    % Need to calculate the post-deformed image, this way I can use it and
    % pre-deformed for MI calculation
    [rows,cols] = size(TumorAreas(:,:,i));
    rows = rows + 1;
    cols = cols+1;
    [z,x] = ndgrid(1:rows,1:cols);
    % Apply the displacements based on FEA results, or FIELD II depending on
    % comments
    % z_new = z + simresult.axial_disp;
    % x_new = x + simresult.lateral_disp;
    z_new = z + AM2D.Axial;
    x_new = x + AM2D.Lateral;
    z_new = min(max(z_new,1),rows);
    x_new = min(max(x_new,1),cols);
    
    z_new = z_new(1:end-1,1:end-1);
    x_new = x_new(1:end-1,1:end-1);
    x = x(1:end-1,1:end-1);
    z = z(1:end-1,1:end-1);
    
    % Change this to subtract the double of eaach image.
    post_deformation_image = interp2(x,z,double(TumorAreas(:,:,i)),x_new,z_new,'linear',0);
    compression_difference_image = post_deformation_image - TumorAreas(:,:,i);
    figure,imshow(compression_difference_image)
    %% Step 4
    % Once it has been done, need to identify image A and image B, and
    % calculate the mutual information value. Because it is a single value,
    % just take note of it, and save it for use in another script.
    mutual_information_values(i) = mutualInfo(uint8(post_deformation_image),uint8(TumorAreas(:,:,i)));
end
%% Save the file, and add it to git.
save('MutualInformationClinicalDataOutputs.mat','mutual_information_values')

!git add MutualInformationClinicalDataOutputs.mat
!git commit -m "Adding  Clinical MI File."
!git push origin
disp('File saved to git.')