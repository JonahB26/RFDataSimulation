% This script is responsible for optimizing the stiffness and thickness of
% the Cooper's ligaments. This will be done by continuously changing the
% values, and calculating the MI value between the two RF frames. Created
% by Jonah Boutin on 11/23/2024.

% Clear the workspace
clear,clc,close all

% %% Mutual Information Dictionary Processing
% 
% % First load in the mutual information data, and get a range of values.
% addpath('C:\Users\jboutin2\Documents\JonahCode\BreastCancerDiagnosis-ML\Data Generation\MutualInformation\')
% load('MutualInformationStatisticsClinicalDataOutputs.mat');
% mean_MI_value = miStats('mean MI value');
% std_MI_value = miStats("std of MI values");
% MI_range = [(mean_MI_value-std_MI_value) (mean_MI_value+std_MI_value)]; %Make the MI range, with some tolerance added
% %% Load the test data and assign thickness/stiffness
% 
% % Now, we must load in the test data
% addpath('C:\Users\jboutin2\Documents\JonahCode\BreastCancerDiagnosis-ML\Data Generation')
% load("OutputDataForTesting.mat")
% test_image = Output.cooperImageData.originalImg; %Grab the actual image (without ligaments)
% tumor_mask = Output.cooperImageData.TumorMask; %Get the tumor mask from the loaded struct
% 
% % figure,imshow(test_image)
% test_image = test_image(80:336,75:331);
% tumor_mask = tumor_mask(80:336,75:331);
% figure,imshow(tumor_mask)
% ligament_thickness_values = [0.25,0.5,0.75,1.0,1.25,1.50,1.75,2.0,2.25,2.50]; %Change the thickness value
% ligament_stiffness_scaling_factor_values = [0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.5]; %Change the scale of  the stiffness for the ligaments
% 
% count = 1; %Initialize counter
% total_count = length(ligament_thickness_values)*length(ligament_stiffness_scaling_factor_values);
%% TEST THE SCRIPT ON THE SIMULATED CLINICAL DATA FROM NIUSHA

load("P39-W0-S2-T.mat");
% load("P97-W1-S2-T.mat");
% 
% 
% bmode_image_pre = load("1_P97-W1-S2.mat_BMODE.mat");
% bmode_image_pre = imresize(double(bmode_image_pre.BMODE),[256,256]);

% load("P97-W1-S2-T.mat");
% Load shit in, essentially the mask represents all the images
tumor_mask = TumorArea;
tumor_mask = imresize(double(tumor_mask),[256,256]);
tumor_mask = tumor_mask > 0.5;
cooper_image = tumor_mask;
test_image = tumor_mask;
[axialres,lateralres] = size(tumor_mask);
%% Run a loop through all possible values
for i = 1%:length(ligament_thickness_values)

    % ligament_thickness = ligament_thickness_values(i);
    ligament_thickness = 0.4475;

    for j = 1%:length(ligament_stiffness_scaling_factor_values)
        % ligament_stiffness_scaling_factor = ligament_stiffness_scaling_factor_values(j);
        ligament_stiffness_scaling_factor = 150000;
        %% Add the ligaments to the image, obtain masks (COMMENTED FOR TEST)
        
        figure;
        imshow(test_image) %Show the image so ligaments can be added
        hold on
        lateralres = size(test_image,2); %This line and next obtains resolution
        axialres = size(test_image,1);
        [cooper_mask,cooper_image] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness); %Add the ligaments

        % % figure,imshow(cooper_mask)
        % 
        % 
        % % Delete the intermediate image for organizational purposes, close the
        % % figure
        % % delete('intermediate_cooper_figure.png');
        % close all%Close figure
        %% Assign stiffness values
        
        test_YM_image = zeros(axialres,lateralres);
        
        % First assign YM to tumor
        % randTumorYM = 5000 + (15000 - 5000)*rand(1);%Generate random YM between 5-15kPa
        randTumorYM = 17000;%Generate random YM between 5-15kPa

        %Set range for tumor YM throughout image
        minTumorYM = randTumorYM - 1500; 
        maxTumorYM = randTumorYM + 1500;
        %Modify the YM image, do this based on counting the true values in the
        %tumor mask
        test_YM_image(tumor_mask) = minTumorYM + (maxTumorYM - minTumorYM)*rand(sum(tumor_mask,'all'),1);
        
        % Then, assign YM to ligaments, do this based on counting the true values
        % in the cooper mask

        % COMMENTED BELOW LINE FOR A TEST
        % test_YM_image(cooper_mask) = ligament_stiffness_scaling_factor*(2500000 + (3500000 - 2500000)*rand(sum(cooper_mask,'all'),1));
        test_YM_image(cooper_mask) = ((0.05*ligament_stiffness_scaling_factor) + (ligament_stiffness_scaling_factor - (0.05*ligament_stiffness_scaling_factor))*rand(sum(cooper_mask,'all'),1));
        % Now, assign YM to the background
        test_YM_image(test_YM_image==0) = 2500 + (3500 - 2500)*rand(sum(test_YM_image==0,"all"),1);
        %% Perform FEA and FIELD II
        
        % i = 1; %This is just a test, then it will be a for-loop**********************************
        
        
        % FEA CODE
        % Define the axial/lateral resolution
        axRes = axialres + 1; %Increment by one for calculation
        latRes = lateralres + 1;
        sim_resolution = [axialres,lateralres]; %Size of simulation
        
        % Define the boundary conditions, these have alread been decided and are in
        % the ManualClinicalSimulation as well
        boundary_conditions = BoundaryConditions(); %Instance of Boundary Conditions class
        %Load my Boundary Conditions just to try a few out
        % boundaries = load('ClinicalBoundariesPostPCA.mat').pca_boundary_conditions_clinical;
        % k = 1;
        % boundary_conditions.top_axial = InterpAndAddPoints(boundaries(k,1:256),257);
        % boundary_conditions.bottom_axial = InterpAndAddPoints(boundaries(k,257:512),257);
        % boundary_conditions.right_axial = InterpAndAddPoints(boundaries(k,513:768),257);
        % boundary_conditions.left_axial = InterpAndAddPoints(boundaries(k,769:1024),257);
        % 
        % boundary_conditions.top_lateral = InterpAndAddPoints(boundaries(k,1025:1280),257);
        % boundary_conditions.bottom_lateral = InterpAndAddPoints(boundaries(k,1281:1536),257);
        % boundary_conditions.right_lateral = InterpAndAddPoints(boundaries(k,1537:1792),257);
        % boundary_conditions.left_lateral = InterpAndAddPoints(boundaries(k,1793:2048),257);
        
        boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
        boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
        boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
        boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
        boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
        boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
        boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
        boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
        % 
        % Define the analysis options, material, and run FEA
        analysis_options = FEMOpts("cartesian", axRes, latRes, "PLANE_STRESS");
        material = Material(test_YM_image, 0.48);
        clc
        disp('Running FEA...')
        % fprintf('On iteration %d of %d\n',count,total_count)
        simresult = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,true);
        clc
        disp('Done Step 2.')
        
        
        % FIELD II CODE
        load("L12-3V.mat") %Load transducer data
        
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
        % I = ones(450,400);
        
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
        % fprintf('On iteration %d of %d\n',count,total_count)
        [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
        
        % Process FIELD II result
        Frame1 = Frame1./max(Frame1(:));
        Frame2 = Frame2./max(Frame2(:));
        if size(Frame1,1) >= size(Frame2,1)
            Frame1 = Frame1(1:size(Frame2,1),:);
        else
            Frame2 = Frame2(1:size(Frame1,1),:);
        end
        
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
        
        clc
        disp('Done Step 3.')
        %% STREAL ON RF
        % % % Frame1 = frame1 ./ max(frame1(:)); %Normalize first frame
% Frame2 = frame2(1:height(frame1),:); %Truncate second frame so they're the same size
% Frame2 = Frame2 ./ Frame2; %Normalize second frame
% Frame1 = frame1;
% Frame2 = frame2;

% Set some arbitrary parameters, use AM2D and STREAL to calculate disps
% Some transducer params, these don't really matter
params.probe.a_t = 1;
params.probe.fc = 5;
params.probe.fs = 50;
params.L = 50;
params.D = 60;

% Frame1 = imresize(Frame1, [2000,256]);
% Frame2 = imresize(Frame2, [2000,256]);

AM2D_disps = RunAM2D(Frame1, Frame2, params);

[Disp_ax,Disp_lat,strainA,strainL,~]...
              = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
              AM2D_disps.Lateral(41:end-60,11:end-10));

% Uncomment this next part for visualization

fig = figure;
subplot(2,2,1)
% norm_ax = (Disp_ax - min(Disp_ax(:))) / (max(Disp_ax(:)) - min(Disp_ax(:)));
imshow(Disp_ax,[])
title("Axial Displacement")
colorbar

subplot(2,2,3)
imshow(Disp_lat,[])
title("Lateral Displacement")
colorbar

subplot(2,2,2)
% norm_strain = (strainA - min(strainA(:))) / (max(strainA(:)) - min(strainA(:)));
imshow(strainA,[])
title("Axial Strain")
colorbar

subplot(2,2,4)
imshow(strainL,[])
title("Lateral Strain")
colorbar
        %% For Testing
        % % Just some code to obtain the deformed image based on the results from
        % % FEA.
        % [rows,cols] = size(cooper_image);
        % rows = rows + 1;
        % cols = cols+1;
        % [z,x] = ndgrid(1:rows,1:cols);
        % % Apply the displacements based on FEA results, or FIELD II depending on
        % % comments
        % % z_new = z + simresult.axial_disp;
        % % x_new = x + simresult.lateral_disp;
        % z_new = z + AM2D.Axial;
        % x_new = x + AM2D.Lateral;
        % z_new = min(max(z_new,1),rows);
        % x_new = min(max(x_new,1),cols);
        % 
        % z_new = z_new(1:end-1,1:end-1);
        % x_new = x_new(1:end-1,1:end-1);
        % x = x(1:end-1,1:end-1);
        % z = z(1:end-1,1:end-1);
        % 
        % test_YM_post = interp2(x,z,double(cooper_image),x_new,z_new,'linear',0);
        % figure,imshow(test_YM_post,[])
        
        %% Mutual Information
        % 
        % MI_value = mutualInfo(uint8(test_YM_post),uint8(cooper_image)); %Calculate MI between the 2 pre & post
        % 
        % MI_converged = false; %First set the check to false
        % 
        % if MI_value > MI_range(1) && MI_value < MI_range(2)
        %     MI_converged = true; %Set the check to true
        % end
        % clc
        
        %% Update the spreadsheet (COMMENTED FOR TEST)
        % Only update if the flag got set to true.

        % % Load in the table if csv already exists, else write to new one
        % if exist('MutualInformation.csv','file')
        %     currentData = readtable('MutualInformation.csv');
        %     currentData(end+1,:) = {MI_value,ligament_thickness,ligament_stiffness_scaling_factor,MI_converged};
        %     writetable(currentData,'MutualInformation.csv')
        % else
        %     MutualInformationValue = MI_value;
        %     LigamentThicknessValue = ligament_thickness;
        %     LigamentStiffnessScalingFactor = ligament_stiffness_scaling_factor;
        %     Convergence = MI_converged;
        %     data = table(MutualInformationValue,LigamentThicknessValue,LigamentStiffnessScalingFactor,Convergence);
        %     writetable(data,'MutualInformation.csv')
        % end
        % count = count + 1; %Increment counter
    end
end

%% Save the file, and add it to git. (COMMENTED FOR TEST)
% 
% !git add MutualInformation.csv
% !git commit -m "Adding  Clinical MI File."
% !git push origin
% disp('File saved to git.')