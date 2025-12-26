%This script is purely to create a slideshow of all my US images, so that I
%know which ones need to be modified using the remove annotations function.

addpath "D:\BreastUSDataOrganizedpre"
addpath "D:\BreastUSDataOrganizedpost"

% %For BrEAST use this folder
%ImageFolder = "D:\BreastUSDatasets\BrEaST Dataset Info\BrEaST-Lesions_USG-images_and_masks";

% %For BUSBRA use this folder
%ImageFolder = "D:\BreastUSDatasets\BUSBRA\Images";

% %For BUSI benign use this folder
%ImageFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\benign";
% 
% %For BUSI malignant use this folder
 %ImageFolder = "D:\BreastUSDatasets\Dataset_BUSI_with_GT\malignant";

% To view the whole post ligament addition dataset use this
ImageFolder = "D:\BreastUSDataOrganizedpost";

ImageFiles = dir(fullfile(ImageFolder, '*.fig'));
ImageFiles = ImageFiles(~[ImageFiles.isdir]);

h = figure;
for i = 2580:length(ImageFiles)
    
    imagePath = fullfile(ImageFolder,ImageFiles(i).name);
    %img = imread(imagePath);
    
   

    %figure(h);
    %imshow(img)
    h = openfig(imagePath);
   

    set(h, 'Name', ImageFiles(i).name, 'NumberTitle', 'off');
    pause(3)
    close(h);
    disp(i)
end