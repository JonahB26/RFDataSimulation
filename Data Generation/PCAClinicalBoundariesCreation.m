% This script is to perform FEA on Niusha's clinical tumor boundaries, and
% then grab the four boundaries, and then run PCA on that array to create a
% set of clinically relevant boundary conditions to make my dataset with.
% Created by Jonah Boutin on 02/01/2025.
% clc
% clear
%% First, run FEA on the data and get the RF frames

% folder_path = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\NiushaTumorBoundaries";%Path to Niusha's files
folder_path = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\NiushaTumorBoundaries';
tumor_files = dir(fullfile(folder_path, '*.mat'));
clinical_tumor_boundaries = zeros([length(tumor_files),(4*256)]);
for i = 1:length(tumor_files)
    tumor_file_path = fullfile(folder_path,tumor_files(i).name);
    % example_file_path = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\NiushaTumorBoundaries\P39-W0-S2-T.mat";
    tumor = load(tumor_file_path);
    tumor_img = imresize(tumor.TumorArea,[256,256]);
        
    % Probably need to adjust this, just copy what I did on lab desktop
    result = GenerateMLData(tumor_img,tumor_img,15000,'malignant');%2nd argument unused anyways
    
    Frame1 = result.Frame1;
    Frame2 = result.Frame2;
    
    %% Now run STREAL on the RF data
    % Set some arbitrary parameters, use AM2D and STREAL to calculate disps
    % Some transducer params, these don't really matter
    params.probe.a_t = 1;
    params.probe.fc = 5;
    params.probe.fs = 50;
    params.L = 50;
    params.D = 60;
    
    AM2D_disps = RunAM2D(Frame1, Frame2, params);
    
    [Disp_ax,Disp_lat,strainA,strainL,~]...
                  = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
                  AM2D_disps.Lateral(41:end-60,11:end-10));
    
    % % Uncomment this next part for visualization
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

    % Now get the inner four boundaries, and interpolate to get 256x256
    top_a = Disp_ax(:,2);
    bottom_a = Disp_ax(:,end-1);
    right_a = Disp_ax(end-1,:)';
    left_a = Disp_ax(2,:)';

    top_l = Disp_lat(:,2)';
    bottom_l = Disp_lat(:,end-1)';
    right_l = Disp_lat(end-1,:)';
    left_l = Disp_lat(2,:)';

    % Now interpolate
    top_axial = InterpAndAddPoints(top_a,256);
    bottom_axial = InterpAndAddPoints(bottom_a,256);
    right_axial = InterpAndAddPoints(right_a,256);
    left_axial = InterpAndAddPoints(left_a,256);

    top_lateral = InterpAndAddPoints(top_l,256);
    bottom_lateral = InterpAndAddPoints(bottom_l,256);
    right_lateral = InterpAndAddPoints(right_l,256);
    left_lateral = InterpAndAddPoints(left_l,256);


    %Now add to the array
    clinical_tumor_boundaries(i,1:256) = top_axial;
    clinical_tumor_boundaries(i,257:512) = bottom_axial;
    clinical_tumor_boundaries(i,513:768) = right_axial;
    clinical_tumor_boundaries(i,769:1024) = left_axial;

    clinical_tumor_boundaries(i,1025:1280) = top_lateral;
    clinical_tumor_boundaries(i,1281:1536) = bottom_lateral;
    clinical_tumor_boundaries(i,1537:1792) = right_lateral;
    clinical_tumor_boundaries(i,1793:2048) = left_lateral;

    fprintf('Finished %d / %d',i,size(clinical_tumor_boundaries,1))
    close all
end
save('ClinicalBoundaries.mat','clinical_tumor_boundaries');
%% Run PCA on that 
boundaries = load('ClinicalBoundaries.mat');
boundaries = boundaries.clinical_tumor_boundaries;

updated_boundaries = boundaries;

[TB_PCA_eigenvectors, TB_PCA_score, TB_PCA_latent, tsquared, explained, mu] = pca(boundaries,'NumComponents',4);

PCA_model.database = TB_PCA_score;
PCA_model.eigenvectors = TB_PCA_eigenvectors;
PCA_model.mu = mu;

% semilogx(cumsum(explained));
% 
% threshold = 95; % Choose 95% variance retention
% num_components = find(cumsum(explained) >= threshold, 1);
% disp(['Optimal number of components: ', num2str(num_components)]);

num_clinical_data = size(updated_boundaries,1);%Number of clinical data points 
num_added_points = 1936 - num_clinical_data;%Number of data points to be added, 1936 is total size
added = zeros(num_added_points,8*256);
updated_boundaries(num_clinical_data+1:num_clinical_data+num_added_points,:) = added;
weights = linspace(0,1,num_added_points);

for i = 1:length(weights)

    selected_data = datasample(TB_PCA_score,2,'Replace',false);

    interpolated = weights(i).*selected_data(1,:) + (1 - weights(i).*selected_data(2,:));
        
    new_data_point = interpolated*TB_PCA_eigenvectors' + mu;

    updated_boundaries(num_clinical_data+i,:) = new_data_point;

    sprintf('Displacement addition number %d has been developed',i)
end
pca_boundary_conditions_clinical = updated_boundaries;
save('ClinicalBoundariesPostPCA.mat','pca_boundary_conditions_clinical')

% !git add .
% !git commit -m "Finished PCA"
% !git push origin
