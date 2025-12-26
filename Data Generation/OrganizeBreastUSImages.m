%This script is just to organize the obtained US images prior to adding
%Cooper's ligaments

addpath "D:\BreastUSDatasets\BrEaST Dataset Info\BrEaST-Lesions_USG-images_and_masks"
addpath "D:\BreastUSDatasets\Dataset_BUSI_with_GT"
addpath "D:\BreastUSDataOrganized"

%% This section is for the BrEAST dataset
BrEASTFolder = "D:\BreastUSDatasets\BrEaST Dataset Info\BrEaST-Lesions_USG-images_and_masks";
outputFolder = "D:\BreastUSDataOrganized";

%Create the output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder)
end

imageFiles = dir(fullfile(BrEASTFolder, '*.png'));
imageFiles = imageFiles(~[imageFiles.isdir]);


filecounter = 1;
for i = 1:length(imageFiles)

    [~, filename, ext] = fileparts(imageFiles(i).name);

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_tumor')
        continue
    end

    newFilename = sprintf('BrEAST_preCooper_%d%s', filecounter, ext);

    BrEASTFile = fullfile(BrEASTFolder, imageFiles(i).name);
    outputFile = fullfile(outputFolder,newFilename);

    copyfile(BrEASTFile,outputFile)
    disp(filecounter)
    filecounter = filecounter + 1;
end

%% This section is for the BUSI 'benign' dataset
BUSIFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\benign";
outputFolder = "D:\BreastUSDataOrganized";

imageFiles = dir(fullfile(BUSIFolder, '*.*'));
imageFiles = imageFiles(~[imageFiles.isdir]);

filecounter = 1;
for i = 1:length(imageFiles)

    [~, filename, ext] = fileparts(imageFiles(i).name);

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_mask') || endsWith(filename,'_mask_1') || endsWith(filename,'_mask_2')
        continue
    end

    newFilename = sprintf('BUSI_benign_preCooper_%d%s', filecounter, ext);

    BUSIFile = fullfile(BUSIFolder, imageFiles(i).name);
    outputFile = fullfile(outputFolder,newFilename);

    copyfile(BUSIFile,outputFile)
    disp(filecounter)
    filecounter = filecounter + 1;
end


%% Malignant
BUSIFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignant";

imageFiles = dir(fullfile(BUSIFolder, '*.*'));
imageFiles = imageFiles(~[imageFiles.isdir]);

filecounter = 1;
for i = 1:length(imageFiles)

    [~, filename, ext] = fileparts(imageFiles(i).name);

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_mask') || endsWith(filename,'_mask_1') || endsWith(filename,'_mask_2')
        continue
    end

    newFilename = sprintf('BUSI_malignant_preCooper_%d%s', filecounter, ext);

    BUSIFile = fullfile(BUSIFolder, imageFiles(i).name);
    outputFile = fullfile(outputFolder,newFilename);

    copyfile(BUSIFile,outputFile)
    disp(filecounter)
    filecounter = filecounter + 1;
end
