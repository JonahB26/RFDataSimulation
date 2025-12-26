%This script will add the Cooper's ligaments to the US images and save them
%accordingly.

addpath "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignantNoAnnotations";

predataFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignantNoAnnotations";
postdataFolder = "D:\BreastUSDataOrganizedpost";

predataFiles = dir(fullfile(predataFolder, '*.fig'));
predataFiles = predataFiles(~[predataFiles.isdir]);

theta = pi/2; %Parameter for Cooper function

counter = 1;
for i = 3:length(predataFiles)
    
    [~, filename, ext] = fileparts(predataFiles(i).name);

    %Check if the filename ends with '_tumor'
    if endsWith(filename,'_mask') || endsWith(filename,'_mask_1') || endsWith(filename,'_mask_2')
        continue
    end
    imagePath = fullfile(predataFolder,predataFiles(i).name);
    fig = openfig(imagePath);
    frame = getframe(fig);
    imageData = frame.cdata;
    grayImage = rgb2gray(imageData);
    imshow(grayImage)
    %img = imread(imagePath);

    % if size(img,3) == 3
    %     rgb2gray(img)
    % end
   %img = double(img);
   %fig = figure;
    %imshow(img)

    hold on

    lateralres = size(grayImage,2);
    axialres = size(grayImage,1);
    AddCoopersLigaments(lateralres,axialres,theta);

    hold off
    %[~,filename,ext] = fileparts(predataFiles(i).name);
    %filename = erase(filename,'preCooper');
    newFilename = sprintf('%sBUSI_postCooper',filename);
    outputPath = fullfile(postdataFolder,newFilename);

    saveas(fig,outputPath);

    %pause(1);

    close(fig);
    sprintf('File number %d has been saved.', counter)
    counter = counter + 1;

end