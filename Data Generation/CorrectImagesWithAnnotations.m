%This script uses the RemoveAnnotations function to correct the annotated
%US images.

%addpath '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/benign';
%addpath '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignant';
addpath '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/BUSBRA/Images'

%addpath "D:\BreastUSDatasets\Dataset_BUSI_with_GT\benign";

%ImageFolder = "D:\BreastUSDatasets\BUSBRA\Images";
%ImageFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/benign';
% ImageFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignant';
ImageFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/BUSBRA/Images';
%ImageFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignant";

ImageFiles = dir(fullfile(ImageFolder, '*.png'));
ImageFiles = ImageFiles(~[ImageFiles.isdir]);

%OutputFolder = "D:\BreastUSDatasets\BUSBRA\ImagesCorrected";
%OutputFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/benignfixednew';
%OutputFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/Dataset_BUSI_with_GT/malignantfixednew';
OutputFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/BreastUSDatasets/BUSBRA/ImagesCorrectednew';
%OutputFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignantNoAnnotations";

counter = 1;
for i = 1:length(ImageFiles)

    [~, filename, ext] = fileparts(ImageFiles(i).name);

    newFilename = sprintf('%s_preCooper.png',filename);
    outputPath = fullfile(OutputFolder,newFilename);

    disp(filename)

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_mask') || endsWith(filename,'_mask_1') || endsWith(filename,'_mask_2')
        continue
    end

    USimagePath = fullfile(ImageFolder,ImageFiles(i).name);

    if exist(outputPath,'file')
        disp(['File ',filename,' exists. Moving on.'])
        continue
    end
    
    %fig = figure;
    k = RemoveAnnotations(USimagePath);


    imwrite(k,outputPath);
    %close(fig);
    counter = counter + 1;
    disp(['Finished file ',num2str(counter)])
end

