% This script is to test out the new versions of Elastosynth
%% FEA
%% Test RunFiniteElementAnalysis
% clc
% clear
% close all
% load("P39-W0-S2-T.mat");
% % BusiBenignImageFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\ClinicalTumorBoundaries-Large';
% % BusiBenignImageFiles = dir(fullfile(BusiBenignImageFolder,"*.mat"));
% % i = 5;
% % ImagePath = fullfile(BusiBenignImageFolder,BusiBenignImageFiles(i).name);
% % load(ImagePath)
% % i = 128;
% % tumor_boundaries_folder = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\NiushaTumorBoundaries";
% % tumor_boundaries = dir(fullfile(tumor_boundaries_folder,"*.mat"));
% % tumor_file = fullfile(tumor_boundaries_folder,tumor_boundaries(i).name);
% % load(tumor_file)
% tumor_mask = TumorArea;
% % tumor_mask = tumor_area;
% tumor_mask = imresize(double(tumor_mask),[256,256]);
% tumor_mask = tumor_mask > 0.5;
% % YM_final = zeros(size(tumor_mask));
% % YM_final(:,:) = 2000 * (1 + (rand(size(YM_final)) - 0.5) * 0.10);
% % YM_final(tumor_mask) = 15000 * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.10);
% 
% % ligament_thickness = 0.4475;
% % ligament_stiffness_scaling_factor = 150000;
% ligament_thickness = 0.4475;
% ligament_stiffness_scaling_factor = 150000;
ligament_thickness = 0.4475;
% ligament_thickness = 1.0;
ligament_stiffness = 150000;
label = 'malignant';
Tumor_YM_middle = 50000;
[simresult,cooper_mask,shape,lateral_boundary_corners,axial_boundary_corners] = FiniteElementAnalysisBatch( ...
    tumor_mask,label,Tumor_YM_middle,ligament_thickness,ligament_stiffness,true);

%% Apoply displacements to the boundaries
mask = tumor_mask + cooper_mask;
lat_disp = ConvertMMToPX(reshape(simresult.lateral_disp, 257, 257));
axial_disp = ConvertMMToPX(reshape(simresult.axial_disp, 257, 257));

% lat_disp = simresult.lateral_disp;
% axial_disp = simresult.axial_disp;

% Resize displacements to match the mask size (if needed)
Disp_ax_resized = imresize(axial_disp, [256, 256], 'bilinear');
Disp_lat_resized = imresize(lat_disp, [256, 256], 'bilinear');

% Create a meshgrid of pixel coordinates
[X, Y] = meshgrid(1:256, 1:256);

% Compute new coordinates using backward warping
X_new = X - Disp_lat_resized;
Y_new = Y - Disp_ax_resized;

% Interpolate the mask at new coordinates
deformed_mask = interp2(double(mask), X_new, Y_new, 'linear', 0);

% Binarize the result
deformed_mask = deformed_mask > 0.5;

% Optional: display
figure;
subplot(2,3,1); imshow(mask); title('Pre-compression Mask');
subplot(2,3,2); imshow(deformed_mask); title('Post-compression Mask');
subplot(2,3,3); imshow(deformed_mask - mask); title('Difference');
subplot(2,3,4); imagesc(Disp_ax_resized); title('Axial Disp (FEA)');colormap('gray');
subplot(2,3,5); imagesc(Disp_lat_resized); title('Lateral Disp (FEA)');colormap('gray');

% 
% % figure;
% % imshow(tumor_mask) %Show the image so ligaments can be added
% % hold on
% % lateralres = size(tumor_mask,2); %This line and next obtains resolution
% % axialres = size(tumor_mask,1);
% % [cooper_mask,cooper_image] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness,'center'); %Add the ligaments
% % close;
% % % YM_final(cooper_mask) = ((0.05*ligament_stiffness_scaling_factor) + (ligament_stiffness_scaling_factor - (0.05*ligament_stiffness_scaling_factor)))*ones(size(YM_final));
% % % YM_final(cooper_mask) = 150000;
% % % YM_final = zeros(size(tumor_mask));
% % % YM_final(:,:) = 2000 * (1 + (rand(size(YM_final)) - 0.5) * 0.10);
% % % YM_final(tumor_mask) = 15000 * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.10);
% % % YM_final(cooper_mask) = 150000 * (1 + (rand(size(YM_final(cooper_mask))) - 0.5) * 0.10);
% % % figure, imshow(YM_final,[0 20000])
% % % First assign YM to tumor
% % label = 'benign';
% % tumor_YM_middle = 4000;
% % if strcmp(label,'malignant')
% %     randTumorYM = randi([(tumor_YM_middle-2000) (tumor_YM_middle+2000)]);%Generate random YM between 5-15kPa
% % elseif strcmp(label,'benign')
% %     randTumorYM = randi([(tumor_YM_middle-1000) (tumor_YM_middle+1000)]);%Generate random YM between 5-15kPa
% % end
% % YM_final = randi([2500 3500],size(cooper_mask));
% % YM_final(tumor_mask) = randTumorYM * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.30);
% % YM_final(cooper_mask) = ligament_stifness * (1 + (rand(size(YM_final(cooper_mask))) - 0.5) * 0.30);
% 
% % Generate test situation
% % 
% % axRes = size(tumor_mask,1) + 1;
% % latRes = size(tumor_mask,2) + 1;
% 
% % axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
% % disp(size(axial_displacements))
% % lateral_displacements = zeros(axRes,latRes);
% % % 
% % % imshow(axial_displacements,[])
% % % colorbar
% 
% %%
% 
% % clc
% % 
% % ConfigureFEM();
% % 
% % % Create material definition
% % material = clib.FEM_Interface.Material_MATLAB;
% % 
% % material.youngs_modulus = flatten(YM_final);
% % material.poissons_ratio = flatten(0.48*ones(size(tumor_mask)));
% % 
% % % Analysis options
% % analysis_options = clib.FEM_Interface.AnalysisOptions;
% % analysis_options.coordinate_system_type = "cartesian";
% % analysis_options.element_type = "PLANE_STRAIN";
% % analysis_options.axial_nodal_resolution = axRes;
% % analysis_options.lateral_nodal_resolution = latRes;
% 
% % Boundary Conditions
% % boundary_conditions = clib.FEM_Interface.BoundaryStruct;
% 
% %% From Displacements
% % % axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
% % % lateral_displacements = zeros(axRes,latRes);
% % 
% % % boundary_conditions.top_axial = axial_displacements(1,:);   
% % % boundary_conditions.bottom_axial = axial_displacements(end,:);
% % % boundary_conditions.left_axial = axial_displacements(:,1)';
% % % boundary_conditions.right_axial = axial_displacements(:,end)';
% % 
% % boundary_conditions.top_lateral = lateral_displacements(1,:);   
% % boundary_conditions.bottom_lateral = lateral_displacements(end,:);
% % boundary_conditions.left_lateral = lateral_displacements(:,1)';
% % boundary_conditions.right_lateral = lateral_displacements(:,end)';
% % 
% % boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
% % boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
% % boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
% % boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
% 
% % % Step 1: Define the four corner values and convert them to pixels
% % top_left = ConvertMMToPX(-(0.07 + (0.09 - 0.07) * rand(1)));
% % top_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));
% % bottom_left = ConvertMMToPX((0.07 + (0.09 - 0.07) * rand(1)));
% % bottom_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));
% % 
% % % Step 2: Generate smooth transitions between corners for each boundary
% % boundary_conditions.left_lateral = linspace(top_left, bottom_left, axRes);
% % boundary_conditions.right_lateral = linspace(top_right, bottom_right, axRes);
% % boundary_conditions.top_lateral = linspace(top_left, top_right, latRes);
% % boundary_conditions.bottom_lateral = linspace(bottom_left, bottom_right, latRes);
% % 
% % left_top = ConvertMMToPX(-(0.22 + (0.25-0.22)*rand(1)));
% % left_bottom = ConvertMMToPX(-(0.03 + (0.07-0.03)*rand(1)));
% % right_top = ConvertMMToPX(-(0.19 + (0.23-0.19)*rand(1)));
% % right_bottom = ConvertMMToPX(-(0.02 + (0.06-0.02)*rand(1)));
% % 
% % boundary_conditions.left_axial = linspace(left_top, left_bottom, axRes);
% % boundary_conditions.right_axial = linspace(right_top, right_bottom, axRes);
% % boundary_conditions.top_axial = linspace(left_top, right_top, latRes);
% % boundary_conditions.bottom_axial = linspace(left_bottom, right_bottom, latRes);
% 
% % boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% % boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% % % boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
% % % boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
% 
% %%
% 
% % tic
% % disp('Running FEA...')
% % simresult = clib.FEM_Interface.RunFiniteElementAnalysis(boundary_conditions,...
% %                                                       size(tumor_mask)+1,...
% %                                                       material, analysis_options);
% % toc
% 
% %%
% % 
% % % THIS NEEDS TO BE UPDATED WITHIN THE FXN
% % axial_disp = ConvertPXToMM(reshape(simresult.axial_displacements.double(), axRes, latRes));
% % lateral_disp = ConvertPXToMM(reshape(simresult.lateral_displacements.double(), axRes, latRes));
% % 
% % axial_strain = -1*(reshape(simresult.axial_strain.double(),axRes-1, latRes-1));
% % lateral_strain = -1*(reshape(simresult.lateral_strain.double(),axRes-1, latRes-1));
% % shear_strain = reshape(simresult.shear_strain.double(),axRes-1, latRes-1);
% % 
% % axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
% % lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);
% 
% % figure
% subplot(2,2,1)
% imshow(axial_disp,[])
% cb = colorbar;
% cb.FontSize = 20;
% title("Axial Displacement (mm)", "FontSize",28)
% 
% 
% subplot(2,2,2)
% imshow(axial_strain,[-0.02 0])
% cb = colorbar;
% cb.FontSize = 20;
% title("Axial Strain", "FontSize",28)
% 
% subplot(2,2,3)
% imshow(lateral_disp,[])
% cb = colorbar;
% cb.FontSize = 20;
% title("Lateral Displacement (mm)", "FontSize",28)
% 
% subplot(2,2,4)
% imshow(lateral_strain,[0 0.01])
% cb = colorbar;
% cb.FontSize = 20;
% title("Lateral Strain", "FontSize",28)

%% FIELD II
% % FIELD II CODE

% % load('L11-5V.mat')
% transducer_list = genTransducers();
% % randomTransducerNum = randi([1 3],1);
% randomTransducerNum = 3;
% transducer = transducer_list{randomTransducerNum};
% sim_resolution = size(cooper_mask);
% 
% % Calculate Image Options
% imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
% 40/1000, 40/1000,10/1000, 256*256,100);
% imageopts.decimation_factor = 2;
% imageopts.axial_FOV = 60/1000;
% imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
% imageopts.slice_thickness = 10/1000;
% 
% field_init(0);
% 
% D = imageopts.axial_FOV;
% L = imageopts.lateral_FOV;
% Z = imageopts.slice_thickness;
% 
% [X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
% % I = ones(220,200);
% % I = ones(450,400);
% I = zeros(256,256); %background
% I(tumor_mask) = 1; %tumor
% I(cooper_mask) = 2; %coopers
% 
% [phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers,'malignant');
% 
% phantom = Phantom(phantom_positions, phantom_amplitudes);
% 
% dispx = interp2(X,Y,axial_disp,phantom_positions(:,1),phantom_positions(:,3));
% dispy = interp2(X,Y,lateral_disp,phantom_positions(:,1),phantom_positions(:,3));
% 
% displacements = zeros(imageopts.n_scatterers, 3);
% displacements(:,3) = dispx/1000;
% displacements(:,1) = dispy/1000;
% displacements(:,2) = 0;
% 
% clc
% disp('Running FIELD II...')
% % fprintf('On iteration %d of %d\n',count,total_count)
% [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
% 
% % Process FIELD II result
% Frame1 = Frame1./max(Frame1(:));
% Frame2 = Frame2./max(Frame2(:));
% if size(Frame1,1) >= size(Frame2,1)
%     Frame1 = Frame1(1:size(Frame2,1),:);
% else
%     Frame2 = Frame2(1:size(Frame1,1),:);
% end
% 
% Frame1 = imresize(Frame1, [2500, 256]);
% Frame2 = imresize(Frame2, [2500, 256]);
output = load('FeaData_1.mat').output;
cooper_mask = output.images.cooper_mask;
tumor_mask = output.images.tumor_mask;
label = output.tumor_info.label;
simresult = struct();
simresult.axial_disp = output.disps.axial_disp;
simresult.lateral_disp = output.disps.lateral_disp;
[Frame1,Frame2,transducer_num] = FieldIIBatch(simresult,cooper_mask,tumor_mask,label);
reconstruction_result = GenerateElastographyImage(Frame1,Frame2,true,[],1);
% figure,imshow(reconstruction_result),title('Elastography Image'),colorbar

% TestFrames = struct();
% TestFrames.Frame1 = Frame1;
% TestFrame.Frame2 = Frame2;
% TestFrames.tumor_mask = tumor_mask;
% TestFrames.cooper_mask = cooper_mask;
% 
% save('TestFrames.mat',"TestFrames")

% params.probe.a_t = 0.63; % frequency dependent attenuation coefficient, in dB/cm/MHz
% params.probe.fc = 5e6; %ultrasound center freq. in MHz
% params.probe.fs = 60; % sampling freq. in MHz
% params.D = 50;
% params.L = 40; 
% 
% 
% AM2D = RunAM2D(Frame1, Frame2, params);
% 
% AM2D.Axial = AM2D.Axial(43:end-53, 11:end-11);
% AM2D.Lateral = AM2D.Lateral(43:end-53, 11:end-11);
% 
% AM2D.Axial = imresize(AM2D.Axial, [axRes, latRes]);
% AM2D.Lateral = imresize(AM2D.Lateral, [axRes, latRes]);
% 
% strainA = imgaussfilt(conv2(AM2D.Axial, [-1 0; 1 0],'valid'),2);
% strainL = imgaussfilt(conv2(AM2D.Lateral, [-1 1; 0 0],'valid'),2);
% 
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

% Set some arbitrary parameters, use AM2D and STREAL to calculate disps
% Some transducer params, these don't really matter
params.probe.a_t = 1;
params.probe.fc = 5;
params.probe.fs = 50;
params.L = 50;
params.D = 60;

Frame1 = imresize(Frame1, [2000,256]);
Frame2 = imresize(Frame2, [2000,256]);

AM2D_disps = RunAM2D(Frame1, Frame2, params);

[Disp_ax,Disp_lat,strainA,strainL,~]...
              = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
              AM2D_disps.Lateral(41:end-60,11:end-10));

% Uncomment this next part for visualization

fig = figure;
subplot(2,2,1)
% figure
% norm_ax = (Disp_ax - min(Disp_ax(:))) / (max(Disp_ax(:)) - min(Disp_ax(:)));
% Disp_ax2 = max(Disp_ax(:)) + min(Disp_ax(:)) - Disp_ax;

imshow(Disp_ax,[])
% imshow(new_Ax,[])
title("Axial Displacement (mm)",FontSize=28)
cb = colorbar;
cb.FontSize = 20;


subplot(2,2,2)
norm_strain = (strainA - min(strainA(:))) / (max(strainA(:)) - min(strainA(:)));
imshow(strainA,[])
title("Axial Strain",FontSize=28)
cb = colorbar;
cb.FontSize = 20;


subplot(2,2,3)
imshow(Disp_lat,[])
title("Lateral Displacement (mm)",FontSize=28)
cb = colorbar;
cb.FontSize = 20;

subplot(2,2,4)
imshow(strainL,[])
title("Lateral Strain",FontSize=28)
cb = colorbar;
cb.FontSize = 20;
% 
% 
% % %% B-mode visualization for pre-deformation (Frame1)
% % % Create the original grid (time samples x scan lines)
% % [original_x, original_y] = meshgrid(1:size(Frame1, 2), 1:size(Frame1, 1));
% % 
% % % Create the target grid (256x256)
% % [target_x, target_y] = meshgrid(linspace(1, size(Frame1, 2), 256), linspace(1, size(Frame1, 1), 256));
% % 
% % % Interpolate the RF data to the target grid
% % rf_data_interpolated = interp2(original_x, original_y, Frame1, target_x, target_y, 'spline');
% % 
% % % Envelope detection using the Hilbert transform
% % envelope = abs(hilbert(rf_data_interpolated)); % Apply Hilbert transform
% % 
% % % Log compression for better visualization
% % log_compressed = 20 * log10(envelope + eps); % Add eps to avoid log(0)
% % 
% % % Normalize the image to its maximum value
% % log_compressed_pre = log_compressed - max(log_compressed(:)); % Normalize to maximum value
% % log_compressed_pre(log_compressed_pre < -40) = -40; % Clip values below -40 dB
% % 
% % % Display the pre-deformation B-mode image
% % figure;
% % subplot(1, 2, 1);
% % imagesc(log_compressed_pre);
% % title('Pre-Deformation B-mode');
% % xlabel('Lateral Position');
% % ylabel('Depth');
% % colormap(gray); % Use grayscale colormap
% % colorbar; % Add a colorbar
% % axis image; % Maintain aspect ratio
% % %% B-mode visualization for pre-deformation (Frame1)
% % % % Create the original grid (time samples x scan lines)
% % % [original_x, original_y] = meshgrid(1:size(Frame1, 2), 1:size(Frame1, 1));
% % % 
% % % % Create the target grid (256x256)
% % % [target_x, target_y] = meshgrid(linspace(1, size(Frame1, 2), 256), linspace(1, size(Frame1, 1), 256));
% % % 
% % % % Interpolate the RF data to the target grid
% % % rf_data_interpolated = interp2(original_x, original_y, Frame1, target_x, target_y, 'spline');
% % % 
% % % % Envelope detection using the Hilbert transform
% % % envelope = abs(hilbert(rf_data_interpolated)); % Apply Hilbert transform
% % % 
% % % % Log compression for better visualization
% % % log_compressed = 20 * log10(envelope + eps); % Add eps to avoid log(0)
% % % 
% % % % Normalize the image to its maximum value
% % % log_compressed_pre = log_compressed - max(log_compressed(:)); % Normalize to maximum value
% % % log_compressed_pre(log_compressed_pre < -40) = -40; % Clip values below -40 dB
% % % 
% % % % Display the pre-deformation B-mode image
% % % figure;
% % % subplot(1, 2, 1);
% % % imagesc(log_compressed_pre);
% % % title('Pre-Deformation B-mode');
% % % xlabel('Lateral Position');
% % % ylabel('Depth');
% % % colormap(gray); % Use grayscale colormap
% % % colorbar; % Add a colorbar
% % % axis image; % Maintain aspect ratio
% % % 
% % % % Extract the lateral (x) and axial (z) coordinates of the displaced scatterers
% % % displaced_x = displaced_positions(:, 1); % Lateral positions
% % % displaced_z = displaced_positions(:, 3); % Axial positions
% % % 
% % % % Create a grid for the displaced scatterer positions
% % % % Define the target grid (256x256)
% % % [target_x_post, target_z_post] = meshgrid(linspace(min(displaced_x), max(displaced_x), 256), ...
% % %                                         linspace(min(displaced_z), max(displaced_z), 256));
% % % 
% % % % Downsample the RF data to match the scatterer positions
% % % % Assuming the RF data is 2500x256, we need to downsample it to 256x256
% % % rf_data_downsampled = imresize(Frame2, [256, 256]); % Downsample using interpolation
% % % 
% % % % Reshape the downsampled RF data to match the scatterer positions
% % % rf_data_reshaped = rf_data_downsampled(:); % Reshape to a column vector
% % % 
% % % % Ensure displaced_x, displaced_z, and rf_data_reshaped have the same length
% % % if length(displaced_x) ~= length(rf_data_reshaped)
% % %     error('Mismatched dimensions: displaced_x, displaced_z, and rf_data_reshaped must have the same length.');
% % % end
% % % 
% % % % Check for NaN values in RF data
% % % if any(isnan(rf_data_reshaped))
% % %     error('RF data contains NaN values.');
% % % end
% % % 
% % % % Interpolate the RF data to the target grid using griddata
% % % rf_data_interpolated_post = griddata(displaced_x, displaced_z, rf_data_reshaped, target_x_post, target_z_post, 'nearest');
% % % 
% % % % Check for NaN values in the interpolated data
% % % if any(isnan(rf_data_interpolated_post(:)))
% % %     warning('Interpolated data contains NaN values. Filling with nearest values.');
% % %     rf_data_interpolated_post = fillmissing(rf_data_interpolated_post, 'nearest');
% % % end
% % % 
% % % % Reshape the interpolated data back to 256x256
% % % rf_data_interpolated_post = reshape(rf_data_interpolated_post, 256, 256);
% % % 
% % % % Envelope detection using the Hilbert transform
% % % envelope_post = abs(hilbert(rf_data_interpolated_post)); % Apply Hilbert transform
% % % 
% % % % Log compression for better visualization
% % % log_compressed_post = 20 * log10(envelope_post + eps); % Add eps to avoid log(0)
% % % 
% % % % Skip normalization since the data is already normalized
% % % % log_compressed_post = log_compressed_post - max(log_compressed_post(:)); % Remove this line
% % % 
% % % % Clip values below -40 dB (optional, adjust as needed)
% % % log_compressed_post(log_compressed_post < -40) = -40;
% % % 
% % % % Display the post-deformation B-mode image
% % % subplot(1, 2, 2);
% % % imagesc(log_compressed_post);
% % % title('Post-Deformation B-mode');
% % % xlabel('Lateral Position');
% % % ylabel('Depth');
% % % colormap(gray); % Use grayscale colormap
% % % colorbar; % Add a colorbar
% % % axis image; % Maintain aspect ratio
