function [result] = GenerateMLData(tumor_mask,ImagePath,tumor_YM_middle,label)
% This function generates the necessary data based on the tumor mask and
% image path. It requires the middle of the tumor YM range, as well as
% either a 'benign' or 'malignant' label. The result is used in ML
% training.
% Created by Jonah Boutin on 12/18/2024
    
%% First define thickness and stiffness for CL
ligament_thickness = 0.4475;
ligament_stifness = 150000;
%% Ligament stuff
% test_image = imread(ImagePath);
% test_image = tumor_mask;
% figure;
% imshow(test_image) %Show the image so ligaments can be added
% hold on
% lateralres = size(test_image,2); %This line and next obtains resolution
% axialres = size(test_image,1);
% [cooper_mask,cooper_image] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness); %Add the ligaments
% close

%% Assign stiffness values

% test_YM_image = zeros(axialres,lateralres);
        
% First assign YM to tumor
if strcmp(label,'malignant')
    randTumorYM = randi([(tumor_YM_middle-2000) (tumor_YM_middle+2000)]);%Generate random YM between 5-15kPa
elseif strcmp(label,'benign')
    randTumorYM = randi([(tumor_YM_middle-1000) (tumor_YM_middle+1000)]);%Generate random YM between 5-15kPa
end


figure;
imshow(tumor_mask) %Show the image so ligaments can be added
hold on
lateralres = size(tumor_mask,2); %This line and next obtains resolution
axialres = size(tumor_mask,1);
[cooper_mask,cooper_image] = AddCoopersLigaments(lateralres,axialres,pi/2,ligament_thickness); %Add the ligaments
close;
% YM_final(cooper_mask) = ((0.05*ligament_stiffness_scaling_factor) + (ligament_stiffness_scaling_factor - (0.05*ligament_stiffness_scaling_factor)))*ones(size(YM_final));
% YM_final(cooper_mask) = 150000;
YM_final = randi([2500 3500],size(cooper_mask));
YM_final(tumor_mask) = randTumorYM * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.30);
YM_final(cooper_mask) = ligament_stifness * (1 + (rand(size(YM_final(cooper_mask))) - 0.5) * 0.30);
% figure, imshow(YM_final,[0 20000])




% Generate test situation

axRes = size(tumor_mask,1) + 1;
latRes = size(tumor_mask,2) + 1;

% axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
% disp(size(axial_displacements))
% lateral_displacements = zeros(axRes,latRes);
% % 
% % imshow(axial_displacements,[])
% % colorbar

%%

clc
ConfigureFEM();

% Create material definition
material = clib.FEM_Interface.Material_MATLAB;

material.youngs_modulus = flatten(YM_final);
material.poissons_ratio = flatten(0.48*ones(size(tumor_mask)));

% Analysis options
analysis_options = clib.FEM_Interface.AnalysisOptions;
analysis_options.coordinate_system_type = "cartesian";
analysis_options.element_type = "PLANE_STRAIN";
analysis_options.axial_nodal_resolution = axRes;
analysis_options.lateral_nodal_resolution = latRes;

% Boundary Conditions
boundary_conditions = clib.FEM_Interface.BoundaryStruct;

%% From Displacements
% % axial_displacements = repmat(linspace(-0.5,0,axRes)',1,latRes);
% % lateral_displacements = zeros(axRes,latRes);
% 
% % boundary_conditions.top_axial = axial_displacements(1,:);   
% % boundary_conditions.bottom_axial = axial_displacements(end,:);
% % boundary_conditions.left_axial = axial_displacements(:,1)';
% % boundary_conditions.right_axial = axial_displacements(:,end)';
% 
% boundary_conditions.top_lateral = lateral_displacements(1,:);   
% boundary_conditions.bottom_lateral = lateral_displacements(end,:);
% boundary_conditions.left_lateral = lateral_displacements(:,1)';
% boundary_conditions.right_lateral = lateral_displacements(:,end)';
% 
% boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
% boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
% boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
% boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);

% Step 1: Define the four corner values and convert them to pixels
top_left = ConvertMMToPX(-(0.07 + (0.09 - 0.07) * rand(1)));
top_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));
bottom_left = ConvertMMToPX((0.07 + (0.09 - 0.07) * rand(1)));
bottom_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));

% Step 2: Generate smooth transitions between corners for each boundary
boundary_conditions.left_lateral = linspace(top_left, bottom_left, axRes);
boundary_conditions.right_lateral = linspace(top_right, bottom_right, axRes);
boundary_conditions.top_lateral = linspace(top_left, top_right, latRes);
boundary_conditions.bottom_lateral = linspace(bottom_left, bottom_right, latRes);

left_top = ConvertMMToPX(-(0.22 + (0.25-0.22)*rand(1)));
left_bottom = ConvertMMToPX(-(0.03 + (0.07-0.03)*rand(1)));
right_top = ConvertMMToPX(-(0.19 + (0.23-0.19)*rand(1)));
right_bottom = ConvertMMToPX(-(0.02 + (0.06-0.02)*rand(1)));

boundary_conditions.left_axial = linspace(left_top, left_bottom, axRes);
boundary_conditions.right_axial = linspace(right_top, right_bottom, axRes);
boundary_conditions.top_axial = linspace(left_top, right_top, latRes);
boundary_conditions.bottom_axial = linspace(left_bottom, right_bottom, latRes);

% boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% % boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
% % boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);

%%

tic

simresult = clib.FEM_Interface.RunFiniteElementAnalysis(boundary_conditions,...
                                                      size(tumor_mask)+1,...
                                                      material, analysis_options);
toc

%%

axial_disp = ConvertPXToMM(reshape(simresult.axial_displacements.double(), axRes, latRes));
lateral_disp = ConvertPXToMM(reshape(simresult.lateral_displacements.double(), axRes, latRes));

% axial_strain = reshape(simresult.axial_strain.double(),axRes-1, latRes-1);
% lateral_strain = reshape(simresult.lateral_strain.double(),axRes-1, latRes-1);
% shear_strain = reshape(simresult.shear_strain.double(),axRes-1, latRes-1);
% 
% axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
% lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);
% 
% figure
% subplot(2,2,1)
% imshow(axial_disp,[])
% colorbar
% title("Axial Displacement (mm)", "FontSize",20)
% 
% subplot(2,2,2)
% imshow(lateral_disp,[])
% colorbar
% title("Lateral Displacement (mm)", "FontSize",20)
% 
% subplot(2,2,3)
% imshow(axial_strain,[-0.02 0.02])
% colorbar
% title("Axial Strain (mm)", "FontSize",20)
% 
% subplot(2,2,4)
% imshow(lateral_strain,[-0.01 0])
% colorbar
% title("Lateral Strain (mm)", "FontSize",20)

%% FIELD II
sim_resolution = size(YM_final);
transducer_list = genTransducers();
randomTransducerNum = randi([1 3],1);
transducer = transducer_list{randomTransducerNum};
% Calculate Image Options
imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
40/1000, 40/1000,10/1000, 256*256,100);
imageopts.decimation_factor = 2;
imageopts.axial_FOV = 60/1000;
imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
imageopts.slice_thickness = 10/1000;

field_init(0);

D = imageopts.axial_FOV;
L = imageopts.lateral_FOV;
Z = imageopts.slice_thickness;

[X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
% I = ones(220,200);
% I = ones(450,400);
I = zeros(256,256); %background
I(tumor_mask) = 1; %tumor
I(cooper_mask) = 2; %coopers

[phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers,'malignant');

phantom = Phantom(phantom_positions, phantom_amplitudes);

dispx = interp2(X,Y,axial_disp,phantom_positions(:,1),phantom_positions(:,3));
dispy = interp2(X,Y,lateral_disp,phantom_positions(:,1),phantom_positions(:,3));

displacements = zeros(imageopts.n_scatterers, 3);
displacements(:,3) = dispx/1000;
displacements(:,1) = dispy/1000;
displacements(:,2) = 0;

clc
disp('Running FIELD II...')
% fprintf('On iteration %d of %d\n',count,total_count)
[Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);


% Process FIELD II result
% Frame1 = Frame1./max(Frame1(:));
% Frame2 = Frame2./max(Frame2(:));


if size(Frame1,1) >= size(Frame2,1)
    Frame1 = Frame1(1:size(Frame2,1),:);
else
    Frame2 = Frame2(1:size(Frame1,1),:);
end

Frame1 = imresize(Frame1, [2500, 256]);
Frame2 = imresize(Frame2, [2500, 256]);

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
% 
% % params.probe.a_t = 0.63; % frequency dependent attenuation coefficient, in dB/cm/MHz
% % params.probe.fc = 5e6; %ultrasound center freq. in MHz
% % params.probe.fs = 60; % sampling freq. in MHz
% % params.D = 50;
% % params.L = 40; 
% % 
% % 
% % AM2D = RunAM2D(Frame1, Frame2, params);
% % 
% % AM2D.Axial = AM2D.Axial(43:end-53, 11:end-11);
% % AM2D.Lateral = AM2D.Lateral(43:end-53, 11:end-11);
% % 
% % AM2D.Axial = imresize(AM2D.Axial, [axRes, latRes]);
% % AM2D.Lateral = imresize(AM2D.Lateral, [axRes, latRes]);
% % 
% % strainA = imgaussfilt(conv2(AM2D.Axial, [-1 0; 1 0],'valid'),2);
% % strainL = imgaussfilt(conv2(AM2D.Lateral, [-1 1; 0 0],'valid'),2);
% % 
% % figure
% % subplot(1,2,1)
% % imshow(strainA,[])
% % title('Axial Strain')
% % colorbar
% % 
% % subplot(1,2,2)
% % imshow(strainL, [])
% % title('Lateral Strain')
% % colorbar
% 
% % Set some arbitrary parameters, use AM2D and STREAL to calculate disps
% % Some transducer params, these don't really matter
% params.probe.a_t = 1;
% params.probe.fc = 5;
% params.probe.fs = 50;
% params.L = 50;
% params.D = 60;
% 
% % Frame1 = imresize(Frame1, [2000,256]);
% % Frame2 = imresize(Frame2, [2000,256]);
% 
% AM2D_disps = RunAM2D(Frame1, Frame2, params);
% 
% [Disp_ax,Disp_lat,strainA,strainL,~]...
%               = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
%               AM2D_disps.Lateral(41:end-60,11:end-10));
% 
% % Uncomment this next part for visualization
% 
% fig = figure;
% subplot(2,2,1)
% % norm_ax = (Disp_ax - min(Disp_ax(:))) / (max(Disp_ax(:)) - min(Disp_ax(:)));
% imshow(Disp_ax,[])
% title("Axial Displacement")
% colorbar
% 
% subplot(2,2,3)
% imshow(Disp_lat,[])
% title("Lateral Displacement")
% colorbar
% 
% subplot(2,2,2)
% % norm_strain = (strainA - min(strainA(:))) / (max(strainA(:)) - min(strainA(:)));
% imshow(strainA,[])
% title("Axial Strain")
% colorbar
% 
% subplot(2,2,4)
% imshow(strainL,[])
% title("Lateral Strain")
% colorbar

reconstruction_result = GenerateElastographyImage(Frame1,Frame2,true);
% 
% % Set range for tumor YM throughout image
% minTumorYM = randTumorYM - 1500; 
% maxTumorYM = randTumorYM + 1500;
% test_YM_Image(tumor_mask) = 15000 * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.10);
% 
% % Assign YM to Cooper's
% test_YM_image(cooper_mask) = ((0.05*ligament_stifness) + (ligament_stifness - (0.05*ligament_stifness))*rand(sum(cooper_mask,'all'),1));
% 
% % Now, assign YM to the background
% test_YM_image(test_YM_image==0) = 2500 + (3500 - 2500)*rand(sum(test_YM_image==0,"all"),1);
% %% Perform FEA and FIELD II
% % FEA CODE
% % Define the axial/lateral resolution
% axRes = axialres + 1; %Increment by one for calculation
% latRes = lateralres + 1;
% sim_resolution = [axialres,lateralres]; %Size of simulation
% 
% % Define the boundary conditions, these have alread been decided and are in
% % the ManualClinicalSimulation as well
% boundary_conditions = BoundaryConditions(); %Instance of Boundary Conditions class
% boundary_conditions.left_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.03 + (0.07-0.03)*rand(1)),axRes);
% boundary_conditions.right_axial = linspace(-(0.19 + (0.23-0.19)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),axRes);
% boundary_conditions.top_axial = linspace(-(0.22 + (0.25-0.22)*rand(1)),-(0.19 + (0.23-0.19)*rand(1)),latRes);
% boundary_conditions.bottom_axial = linspace(-(0.03 + (0.07-0.03)*rand(1)),-(0.02 + (0.06-0.02)*rand(1)),latRes);
% boundary_conditions.top_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% boundary_conditions.bottom_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),0.09 + (0.11 - 0.09)*rand(1),latRes);
% boundary_conditions.left_lateral = linspace(-(0.07 + (0.09 - 0.07)*rand(1)),-(0.07 + (0.09 - 0.07)*rand(1)),axRes);
% boundary_conditions.right_lateral = linspace(0.09 + (0.11 - 0.09)*rand(1),0.09 + (0.11 - 0.09)*rand(1),axRes);
% 
% % Define the analysis options, material, and run FEA
% analysis_options = FEMOpts("cartesian", axRes, latRes, "PLANE_STRESS");
% material = Material(test_YM_image, 0.48);
% clc
% disp('Running FEA...')
% simresult = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,false);
% clc
% disp('Done Step 2.')
% 
% 
% % FIELD II CODE
% transducer_list = genTransducers();
% transducer = transducer_list{randi([1 3],1)}; %Randomly pick a transducer
% 
% % Calculate Image Options
% imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
% 40/1000, 40/1000,10/1000, 10e4,100);
% imageopts.decimation_factor = 2;
% imageopts.axial_FOV = 60/1000;
% imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
% imageopts.slice_thickness = 10/1000;
% 
% field_init();
% 
% D = imageopts.axial_FOV;
% L = imageopts.lateral_FOV;
% Z = imageopts.slice_thickness;
% 
% [X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
% I = ones(220,200);
% 
% [phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers);
% 
% phantom = Phantom(phantom_positions, phantom_amplitudes);
% 
% dispx = interp2(X,Y,simresult.axial_disp,phantom_positions(:,1),phantom_positions(:,3));
% dispy = interp2(X,Y,simresult.lateral_disp,phantom_positions(:,1),phantom_positions(:,3));
% 
% displacements = zeros(imageopts.n_scatterers, 3);
% displacements(:,3) = dispx/1000;
% displacements(:,1) = dispy/1000;
% displacements(:,2) = 0;
% 
% clc
% disp('Running FIELD II...')
% [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);
% 
% % Process FIELD II result
% % Frame1 = Frame1./max(Frame1(:));
% % Frame2 = Frame2./max(Frame2(:));
% 
% 
% if size(Frame1,1) >= size(Frame2,1)
%     Frame1 = Frame1(1:size(Frame2,1),:);
% else
%     Frame2 = Frame2(1:size(Frame1,1),:);
% end
% 
% Frame1 = imresize(Frame1, [2500, 256]);
% Frame2 = imresize(Frame2, [2500, 256]);
% %% Save result

% reconstruction_result = GenerateElastographyImage(Frame1,Frame2,true);

result = struct();
result.Frame1 = Frame1;
result.Frame2 = Frame2;
result.transducer_info = transducer;


% result.boundary_conditions = struct();
% result.boundary_conditions.left_axial = boundary_conditions.left_axial.double();
% result.boundary_conditions.right_axial = boundary_conditions.right_axial.double();
% result.boundary_conditions.bottom_axial = boundary_conditions.bottom_axial.double();
% result.boundary_conditions.top_axial = boundary_conditions.top_axial.double();
% 
% result.boundary_conditions.left_lateral = boundary_conditions.left_lateral.double();
% result.boundary_conditions.right_axial = boundary_conditions.right_lateral.double();
% result.boundary_conditions.bottom_lateral = boundary_conditions.bottom_lateral.double();
% result.boundary_conditions.top_lateral = boundary_conditions.top_lateral.double();

result.image_information.bmode_uncompressed = tumor_mask;
result.image_information.cooper_image = cooper_image;
result.image_information.tumor_mask = tumor_mask;
result.image_information.cooper_mask = cooper_mask;
result.image_information.YM_image = YM_final;

result.elastography_image = reconstruction_result;

% NEED TO DISTINGUISH BETWEEN MALIGNANT/BENIGN
result.label = label;
