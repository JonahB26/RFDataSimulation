% This script is responsible for the optimization of the thickness and
% stiffness of the Cooper's ligaments in the ultrasound images. This will
% be accomplished by first calculating the deformation between clinical RF
% frames from Niusha's files, and then using those in an optimization
% algorithm to get optimal thickness and stiffness values.
% Created by Jonah Boutin on Nov 27th, 2024
%% Define ligament thickness and stiffness parameters
% clc,clear,close all %Clean workspace

% ligament_thickness = 1.0;
% ligament_stiffness_scaling_factor = 0.1;
%% Optimization Parameters
clc,clear,close all

% Initial values
% initial_values = [1.0,0.1]; %[lig thickness,lig stiffness scaling factor]
initial_values = [0.5,150000]; %[lig thickness,lig stiffness scaling factor]

% Bounds
% lb = [0.1,0.001]; %Lower bounds
% ub = [2.0,1.0]; %Upper bounds
lb = [0.1,30000]; %Lower bounds
ub = [2.0,3000000]; %Upper bounds

% Objective function
objective_function = @(x) calculate_loss_function(x(1),x(2));

% Optimization options
options = optimoptions("fmincon",'Display','iter','Algorithm','active-set');

% Run optimization
[x_opt,fval] = fmincon(objective_function,initial_values,[],[],[],[],lb,ub,[],options);

% Display results 
disp('Optimal Ligament Thickness:') 
disp(x_opt(1)) 
disp('Optimal Ligament Stiffness Scaling Factor:') 
disp(x_opt(2)) 
disp('Optimal Loss Function Value:') 
disp(fval)

function loss = calculate_loss_function(ligament_thickness,ligament_stiffness_scaling_factor)
%% First, load in the clinical B-mode image, as well as assign the clinical RF frames

% First load in clinical RF data.
frame_name = "P97-W1-S2.mat";
% frame_name = "P94-W0-S2.mat";
rf = load(frame_name);
rf = rf.rf1;

% According to Niusha's metadata, located in Data/Breast/Metadata in
% onedrive data, the pre-compression for this data is frame 8, and the post
% compression for this data is frame 13.
frame1 = rf(:,:,8);
frame2 = rf(:,:,18);

% Now, load in the b-mode image data pertaining to this case.
tumor_mask = load("P97-W1-S2-T.mat");
tumor_mask = imresize(double(tumor_mask.TumorArea),[256,256]);
tumor_mask = logical(tumor_mask);

bmode_image_pre = load("1_P97-W1-S2.mat_BMODE.mat");
bmode_image_pre = imresize(double(bmode_image_pre.BMODE),[256,256]);

% Get the undeformed B-Mode image
rf_envelope = abs(hilbert(frame1));
rf_log_compressed = 20*log10(rf_envelope + 1);
rf_normalized = mat2gray(rf_log_compressed);
bmode_image_pre = uint8(rf_normalized*255);
bmode_image_pre = uint8(imresize(double(bmode_image_pre),[256,256]));

% Get the deformed B-Mode image
rf_envelope = abs(hilbert(frame2));
rf_log_compressed = 20*log10(rf_envelope + 1);
rf_normalized = mat2gray(rf_log_compressed);
bmode_image_post = uint8(rf_normalized*255);
bmode_image_post = uint8(imresize(double(bmode_image_post),[256,256]));

% % AS A TEST CALCULATE B-MODE FROM FIRST FRAME
% rf_envelope2 = abs(hilbert(frame1));
% rf_log_compressed2 = 20*log10(rf_envelope2 + 1);
% rf_normalized2 = mat2gray(rf_log_compressed2);
% bmode_image_pre = uint8(rf_normalized2*255);
% bmode_image_pre = imresize(double(bmode_image_pre),[256,256]);

% Visualize the pre bmode and post bmode images
% figure,subplot(1,2,1),imshow(bmode_image_pre,[]),subplot(1,2,2),imshow(bmode_image_post,[])

%% Second, run STREAL/AM2D on the two clinical RF frames
% % % Frame1 = frame1 ./ max(frame1(:)); %Normalize first frame
% % % Frame2 = frame2(1:height(frame1),:); %Truncate second frame so they're the same size
% % % Frame2 = Frame2 ./ Frame2; %Normalize second frame
% Frame1 = frame1;
% Frame2 = frame2;
% 
% % Set some arbitrary parameters, use AM2D and STREAL to calculate disps
% % Some transducer params, these don't really matter
% params.probe.a_t = 1;
% params.probe.fc = 5;
% params.probe.fs = 50;
% params.L = 50;
% params.D = 60;
% 
% Frame1 = imresize(Frame1, [2000,256]);
% Frame2 = imresize(Frame2, [2000,256]);
% 
% AM2D_disps = RunAM2D(Frame1, Frame2, params);
% 
% [Disp_ax,Disp_lat,~,~,~]...
%               = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
%               AM2D_disps.Lateral(41:end-60,11:end-10));
% 
% % % Uncomment this next part for visualization
% % 
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

%% Third, add ligaments to my clinical B-mode image, using defined thi

[axialres,lateralres] = size(bmode_image_pre); %Calculate resolution

figure;
imshow(bmode_image_pre) %Show the image so ligaments can be added
hold on

[cooper_mask,cooper_image] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness); %Add the ligaments
close %Close figure for organization
% load('CooperMask.mat'); %Load saved cooper data
%% Extra boundary conditions based on results in the second step
load('SimulatedBoundaryConditions.mat');
boundary1_conditions = BoundaryConditions(); %Initiate class
% 
% boundary_conditions.top_axial = Disp_ax(5,:); %5th row in axial
% boundary_conditions.top_axial = 0.01*InterpAndAddPoints(boundary_conditions.top_axial,257);
% 
% boundary_conditions.bottom_axial = Disp_ax(end-5,:); %5th last row in axial
% boundary_conditions.bottom_axial = 0.01*InterpAndAddPoints(boundary_conditions.bottom_axial,257);
% 
% boundary_conditions.left_axial = Disp_ax(:,5); %5th column in axial
% boundary_conditions.left_axial = 0.01*InterpAndAddPoints(boundary_conditions.left_axial,257);
% 
% boundary_conditions.right_axial = Disp_ax(:,end-5); %5th last column in axial
% boundary_conditions.right_axial = 0.01*InterpAndAddPoints(boundary_conditions.right_axial,257);
% 
% 
% boundary_conditions.top_lateral = Disp_lat(5,:); %5th row in lateral
% boundary_conditions.top_lateral = 0.01*InterpAndAddPoints(boundary_conditions.top_lateral,257);
% 
% boundary_conditions.bottom_lateral = Disp_lat(end-5,:); %5th last row in lateral
% boundary_conditions.bottom_lateral = 0.01*InterpAndAddPoints(boundary_conditions.bottom_lateral,257);
% 
% boundary_conditions.left_lateral = Disp_lat(:,5); %5th column in lateral
% boundary_conditions.left_lateral = 0.01*InterpAndAddPoints(boundary_conditions.left_lateral,257);
% 
% boundary_conditions.right_lateral = Disp_lat(:,end-5); %5th last column in lateral
% boundary_conditions.right_lateral = 0.01*InterpAndAddPoints(boundary_conditions.right_lateral,257);
[axRes,latRes] = size(bmode_image_pre); %Get resolution
axRes = axRes + 1;
latRes = latRes + 1;
% boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
% boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
% boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
% boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
% boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
% boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);

boundary1_conditions.left_axial = boundary_conditions.left_axial;
boundary1_conditions.right_axial = boundary_conditions.right_axial;
boundary1_conditions.top_axial = boundary_conditions.top_axial;
boundary1_conditions.bottom_axial = boundary_conditions.bottom_axial;
boundary1_conditions.top_lateral = boundary_conditions.top_lateral;
boundary1_conditions.bottom_lateral = boundary_conditions.bottom_lateral;
boundary1_conditions.left_lateral = boundary_conditions.left_lateral;
boundary1_conditions.right_lateral = boundary_conditions.right_lateral;

%% Assign stiffness values
[axRes,latRes] = size(bmode_image_pre); %Get resolution
load('YMmat.mat')
% load("cooper_stiffness.mat")
% test_YM_image(cooper_mask) = ligament_stiffness_scaling_factor*cooper_stiffness;
test_YM_image(cooper_mask) = ((0.05*ligament_stiffness_scaling_factor) + (ligament_stiffness_scaling_factor - (0.05*ligament_stiffness_scaling_factor))*rand(nnz(cooper_mask),1));

% test_YM_image = zeros(axRes,latRes);
% 
% % COMMENTING TO KEEP TUMOR YM CONSTANT FOR OPTIMIZATION
% % % First assign YM to tumor
% randTumorYM = 15000;%5000 + (15000 - 5000)*rand(1);%Generate random YM between 5-15kPa
% % %Set range for tumor YM throughout image
% minTumorYM = randTumorYM - 1500; 
% maxTumorYM = randTumorYM + 1500;
% %Modify the YM image, do this based on counting the true values in the
% %tumor mask
% test_YM_image(tumor_mask) = minTumorYM + (maxTumorYM - minTumorYM)*rand(nnz(tumor_mask),1);
% 
% %Add YM to cooper mask
% test_YM_image(cooper_mask) = ligament_stiffness_scaling_factor*(2500000 + (3500000 - 2500000)*rand(nnz(cooper_mask),1));
% 
% %Add YM to bakckground
% test_YM_image(test_YM_image==0) = 2500 + (3500 - 2500)*rand(sum(test_YM_image==0,"all"),1);
% % load("YMOptimization.mat"); %Load saved YM matrix
%% Perform FEA

% Define the analysis options, material, and run FEA
analysis_options = FEMOpts("cartesian", axRes+1, latRes+1, "PLANE_STRESS");
material = Material(test_YM_image, 0.48);

disp('Running FEA...')
simresult = RunFiniteElementAnalysis(analysis_options, material, boundary1_conditions,false);
clc
%% Reconstruct deformed image using results from FEA, define loss function
% Just some code to obtain the deformed image based on the results from
% FEA.
% NEED TO GET DEFORMATION INFO FROM MATTHEW
% % NEED TO KNOW DPI OF SCREEN (Run: 'wmic desktopmonitor get
% % PixelsPerXLogicalInch' in bash terminal to obtain)
% dpi_laptop_Jonah = 144;
dpi_lab_desktop = 96;
dpi = dpi_lab_desktop;
bmode_cooper_post = ApplyDeformation(bmode_image_pre,simresult.axial_disp,simresult.lateral_disp,true,dpi);%Set is_mm bool to true so it converts the mm displacements to pixel
bmode_clinical_post = ApplyDeformation(bmode_image_post,simresult.axial_disp,simresult.lateral_disp,true,dpi);
fprintf('Max_clinical_post: %f, Min_clinical_post: %f',max(bmode_clinical_post(:)),min(bmode_clinical_post(:)));


% % % FOR TEST
% 
% % Convert matrix to mm
% d_ax_mm = simresult.axial_disp*100;
% d_lat_mm = simresult.lateral_disp*100;
% 
% % Convert to pixels
% d_ax = (dpi/25.4).*d_ax_mm;
% d_lat = (dpi/25.4).*d_lat_mm;
% 
% [rows,cols] = size(bmode_image_pre);
% rows = rows + 1;
% cols = cols+1;
% [z,x] = ndgrid(1:rows,1:cols);
% 
% % Apply the displacements based on FEA results
% % z_new = z + simresult.axial_disp;
% % x_new = x + simresult.lateral_disp;
% z_new = z + d_ax;
% x_new = x + d_lat;
% 
% z_new = min(max(z_new,1),rows);
% x_new = min(max(x_new,1),cols);
% 
% z_new = z_new(1:end-1,1:end-1);
% x_new = x_new(1:end-1,1:end-1);
% x = x(1:end-1,1:end-1);
% z = z(1:end-1,1:end-1);
% % 
% bmode_cooper_post = interp2(x,z,double(bmode_image_pre),x_new,z_new,'linear',0);
% bmode_clinical_post = interp2(x,z,double(bmode_image_post),x_new,z_new,'linear',0);

% bmode_cooper_post = griddata(x,z,double(bmode_image_pre),x_new,z_new);
% bmode_cooper_post = uint8(bmode_cooper_post);
% figure,subplot(1,2,1),imshow(bmode_clinical_post,[]),title('Clinical Deformed B-mode Image'),subplot(1,2,2),imshow(bmode_cooper_post,[]),title('Clinical Deformed B-mode Image (With cooper FEA disps)')
%% LOSS FUNCTION
% % To visualize
% figure
% suplot(1,2,1),imshow(cooper_image),title('Pre-deformed')
% subplot(1,2,2),imshow(cooper_post),title('Post-deformed')

loss = mean((bmode_clinical_post-double(bmode_image_pre)).^2,'all');
% diff = double(bmode_image_post) - double(cooper_post);
% figure,imshow(diff)
close all
end