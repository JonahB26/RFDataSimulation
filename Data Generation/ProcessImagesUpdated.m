%This script reruns the Cooper's ligament addition to the data, there were
%issues in the current data that caused the images to not resemble the
%original data, and this script addresses that by properly sizing the
%image.

%addpath('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignantNoAnnotations/')
addpath('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/benignfixednew');
addpath('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/')
addpath('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/BUSBRA/ImagesCorrectednew');
addpath('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignantfixednew');

predataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/benignfixednew';
%predataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignantfixednew';
%predataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/BUSBRA/ImagesCorrectednew';
predataFiles = dir(fullfile(predataFolder, '*.png'));
predataFiles = predataFiles(~[predataFiles.isdir]);
% postrawdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSIBenignPostCooperRawFilesUpdated';%Addpath from desktop
% postmatdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSIBenignPostCooperMatFilesUpdated';
% postrawdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSIMalignantPostCooperRawFilesUpdated';
% postmatdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSIMalignantPostCooperMatFilesUpdated';
%postrawdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSBRAPostCooperRawFilesUpdated1';
%postmatdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/BUSBRAPostCooperMatFilesUpdated1';
postrawdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/BreastCancerDiagnosis-ML/TestFilesRaw';
postmatdataFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/BreastCancerDiagnosis-ML/TestFilesMat';

theta = pi/2; %Parameter for Cooper function

counter = 1;
for i = 323%:length(predataFiles)
    
    [~, filename, ext] = fileparts(predataFiles(i).name);

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_mask') || endsWith(filename,'_mask_1') || endsWith(filename,'_mask_2')
        continue
    end

    imagePath = fullfile(predataFolder,predataFiles(i).name);
    % fig = openfig(imagePath);
    % frame = getframe(fig);
    % img = frame.cdata;
    %grayImage = rgb2gray(imageData);
    %imshow(grayImage)

    img = imread(imagePath);

    if size(img,3) == 3
        img = rgb2gray(img);
    end

    figure;
    imshow(img)

    hold on

    lateralres = size(img,2);
    axialres = size(img,1);

    AddCoopersLigaments(lateralres,theta);
    hold off

    newfilename = [filename,'_075thickness_raw.png'];

    outputhPath = fullfile(postrawdataFolder,newfilename);

    exportgraphics(gcf,outputhPath,'Resolution',300,'BackgroundColor','none');
    
    loaded = imread(outputhPath);

    finalname = [filename,'_075_thickness'];

    finalPath = fullfile(postmatdataFolder,finalname);

    if size(loaded,3) == 3
        loaded = rgb2gray(loaded);
    end

    loaded = double(loaded);
    loaded = imresize(loaded, [axialres lateralres]);%NEEDS TO BE AXIAL/LATERAL NOT LATERAL/AXIAL
    loaded = uint8(loaded);

    cooperImageData.originalImg = img;
    cooperImageData.cooperImg = loaded;
    cooperMask = loaded - uint8(img);
    cooperMask = cooperMask > 50;
    cooperImageData.cooperMask = cooperMask;

    save(finalPath,'cooperImageData');

    sprintf('File num %d',i)

    % close all

end
