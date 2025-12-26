% This script is intended to run PCA ON Niusha's clinical tumor boundaries,
% to generate a larger amount for v1.0 of my dataset. 
% Created by Jonah Boutin on 01/20/2024.
clc
clear
boundary_folder = "C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\NiushaTumorBoundaries";
% addpath("C:\Users\1bout\OneDrive\SamaniLab\MESc Files\ClinicalTumorBoundaries-Final")
% boundary_folder =
% "/home/deeplearningtower/Documents/MATLAB/JonahCode/NiushaTumorBoundaries;
boundary_files = dir(fullfile(boundary_folder,'*.mat'));
tumor_boundaries_array = zeros(length(boundary_files),65536); %Array to hold boundaries during PCA (num elements in a flattened 256x256 is 65536)

clinical_boundaries_folder = "C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\ClinicalTumorBoundaries-Large/";


for i = 1:length(boundary_files)
    tumor_boundary = load(fullfile(boundary_folder,boundary_files(i).name));
    tumor_area = tumor_boundary.TumorArea;

    % Save to folder
    tumor_area = imresize(double(tumor_area),[256,256],'nearest');
    tumor_area = tumor_area > 0.5;
    % file_hex = DataHash(tumor_area, 'array','hex');
    name = strcat("Tumor_",num2str(i));
    filename = clinical_boundaries_folder+name+".mat";
    save(char(filename), "tumor_area");

    % Add to array
    tumor_area_flat = double(tumor_area(:)');
    tumor_boundaries_array(i,:) = tumor_area_flat;
end




[TB_PCA_eigenvectors, TB_PCA_score, TB_PCA_latent, tsquared, explained, mu] = pca(tumor_boundaries_array,'NumComponents',101);

% numComponents explains how many components are necessary to keep dataset
% variance
% cumulativeVariance = cumsum(explained);
% numComponents = find(cumulativeVariance >= 95, 1); % For 95% variance
% PCA_model.database = TB_PCA_score;
% PCA_model.eigenvectors = TB_PCA_eigenvectors;
% PCA_model.mu = mu;
% 
% save(    '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/PCA_model.mat',"PCA_model")

% semilogx(cumsum(explained));

numAddedPoints = 4500;%Number of data points to be added
weights = linspace(0,1,numAddedPoints);

for i = 1:length(weights)

    selectedData = datasample(TB_PCA_score,2,'Replace',false);

    interpolated = weights(i).*selectedData(1,:) + (1 - weights(i)).*selectedData(2,:);

    added_boundary = interpolated*TB_PCA_eigenvectors' + mu;

    added_boundary = reshape(added_boundary,[256,256]) > 0.5;

   added_boundary = bwareaopen(imclose(added_boundary, strel('disk', 5)), 60); % Clean mask


    tumor_area = added_boundary;

    % file_hex = DataHash(tumor_area, 'array','hex');
    name = strcat("Tumor_",num2str(i + length(boundary_files)));
    filename = clinical_boundaries_folder+name+".mat";
    save(char(filename), "tumor_area");

    sprintf('Displacement addition number %d has been developed',i)
end