function [output,cooper_mask,shape,lateral_boundary_corners,axial_boundary_corners,YM_final] = FiniteElementAnalysisBatch(tumor_mask,label,tumor_YM_middle,ligament_thickness,ligament_stiffness,visualize)
%% Cooper's
close all;
figure;
imshow(tumor_mask) %Show the image so ligaments can be added
hold on
lateralres = size(tumor_mask,2); %This line and next obtains resolution
axialres = size(tumor_mask,1);

[cooper_mask,~] = AddCoopersLigaments(tumor_mask,lateralres,axialres,pi/2,ligament_thickness,'top-right'); %Add the ligaments


% First assign YM to tumor
if strcmp(label,'malignant')
    randTumorYM = randi([(tumor_YM_middle-2000) (tumor_YM_middle+2000)]);%Generate random YM between 5-15kPa
elseif strcmp(label,'benign')
    randTumorYM = randi([(tumor_YM_middle-1000) (tumor_YM_middle+1000)]);%Generate random YM between 5-15kPa
end
% YM_final = randi([2500 3500],size(cooper_mask));
YM_final = randi([9500 10500],size(cooper_mask));
YM_final(tumor_mask) = randTumorYM * (1 + (rand(size(YM_final(tumor_mask))) - 0.5) * 0.30);

YM_final(cooper_mask) = ligament_stiffness * (1 + (rand(size(YM_final(cooper_mask))) - 0.5) * 0.30);
figure;imshow(tumor_mask + cooper_mask);


axRes = size(tumor_mask,1) + 1;
latRes = size(tumor_mask,2) + 1;
%% FEA
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

% Step 1: Define the four corner values and convert them to pixels
top_left = ConvertMMToPX(-(0.07 + (0.09 - 0.07) * rand(1)));
top_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));
bottom_left = ConvertMMToPX((0.07 + (0.09 - 0.07) * rand(1)));
bottom_right = ConvertMMToPX((0.09 + (0.11 - 0.09) * rand(1)));

axial_boundary_corners = [bottom_left bottom_right top_left top_right];

% Step 2: Generate smooth transitions between corners for each boundary
boundary_conditions.left_lateral = linspace(top_left, bottom_left, axRes);
boundary_conditions.right_lateral = linspace(top_right, bottom_right, axRes);
boundary_conditions.top_lateral = linspace(top_left, top_right, latRes);
boundary_conditions.bottom_lateral = linspace(bottom_left, bottom_right, latRes);

left_top = ConvertMMToPX(-(0.22 + (0.25-0.22)*rand(1)));
left_bottom = ConvertMMToPX(-(0.03 + (0.07-0.03)*rand(1)));
right_top = ConvertMMToPX(-(0.19 + (0.23-0.19)*rand(1)));
right_bottom = ConvertMMToPX(-(0.02 + (0.06-0.02)*rand(1)));

lateral_boundary_corners = [left_bottom right_bottom left_top right_top];

boundary_conditions.left_axial = linspace(left_top, left_bottom, axRes);
boundary_conditions.right_axial = linspace(right_top, right_bottom, axRes);
boundary_conditions.top_axial = linspace(left_top, right_top, latRes);
boundary_conditions.bottom_axial = linspace(left_bottom, right_bottom, latRes);

shape = [axRes latRes];

tic
disp('Running FEA...')
simresult = clib.FEM_Interface.RunFiniteElementAnalysis(boundary_conditions,...
                                                      size(tumor_mask)+1,...
                                                      material, analysis_options);
toc

output = struct();

output.axial_disp = ConvertPXToMM(reshape(simresult.axial_displacements.double(), axRes, latRes));
output.lateral_disp = ConvertPXToMM(reshape(simresult.lateral_displacements.double(), axRes, latRes));

output.axial_strain = -1*(reshape(simresult.axial_strain.double(),axRes-1, latRes-1));
output.lateral_strain = -1*(reshape(simresult.lateral_strain.double(),axRes-1, latRes-1));
output.shear_strain = reshape(simresult.shear_strain.double(),axRes-1, latRes-1);

output.axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
output.lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);

if visualize
    figure
    subplot(2,2,1)
    imshow(output.axial_disp,[])
    cb = colorbar;
    cb.FontSize = 20;
    title("Axial Displacement (mm)", "FontSize",28)
    
    
    subplot(2,2,2)
    imshow(output.axial_strain,[-0.02 0])
    cb = colorbar;
    cb.FontSize = 20;
    title("Axial Strain", "FontSize",28)
    
    subplot(2,2,3)
    imshow(output.lateral_disp,[])
    cb = colorbar;
    cb.FontSize = 20;
    title("Lateral Displacement (mm)", "FontSize",28)
    
    subplot(2,2,4)
    imshow(output.lateral_strain,[0 0.01])
    cb = colorbar;
    cb.FontSize = 20;
    title("Lateral Strain", "FontSize",28)

end