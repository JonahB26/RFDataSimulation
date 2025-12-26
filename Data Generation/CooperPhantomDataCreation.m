%This Script is practice to adding Cooper's ligaments to tissue mimicking
%phantoms generated from Matthew's Elastosynth functions
addpath C:/Users/1bout/OneDrive/SamaniLab/Elastosynth/Simulator
%addpath C:/Users/1bout/OneDrive/SamaniLab/MESc Files/CoopersLigamentTestPhantoms

% %Quick for-loop to generate examples
% outputFolder = 'C:/Users/1bout/OneDrive/SamaniLab/MESc Files/CoopersLigamentTestPhantoms';
% if ~exist(outputFolder,'dir')
%     mkdir(outputFolder);
% end
% 
% filename = fullfile(outputFolder,['CooperPhantomTest_' num2str(figurenumber) '.fig']);
% 
% counter = 1;
% while counter <= 15
% 
% clearvars -except counter outputFolder
% clc

clear
clc

%Set the axial and lateral resolution of the phantom
axialres = 500;
lateralres = 500;

[inclusion_locations,success]=GenerateRandomInclusionLocations(2,[50 75] ...
    ,[axialres lateralres],false,false,3,15);

%Set the minimum and maximum Young's modulus values of the inclusions in
%the phantoms, used for randomization
min_inclusion_YM = 0;
max_inclusion_YM = 10;

%Randomize the Young's Modulus of both inclusions
YM1 = min_inclusion_YM + (max_inclusion_YM-min_inclusion_YM)*rand;
YM2 = min_inclusion_YM + (max_inclusion_YM-min_inclusion_YM)*rand;

[YM_img,inclusion_masks]=GeneratePhantom(2,[100;75],inclusion_locations,[YM1;YM2],5,[axialres lateralres]);
[YM_out]=AddPhantomHeterogeneity(YM_img,inclusion_masks,[1.2;0.4;0.4],200);

final = YM_out/max(YM_out,[],'all');
imshow(final)

theta = pi/2; %Angle for rotation matrix in AddCoopersLigaments function, in radians

[x, y, x_r, y_r] = AddCoopersLigaments(lateralres,theta);

% 
% % 
% % filename = fullfile(outputFolder,['CooperPhantomTest_' num2str(counter) '.fig']);
% % 
% % savefig(filename)
% % 
% % counter = counter + 1;
% % end
% % disp('All figures saved.')