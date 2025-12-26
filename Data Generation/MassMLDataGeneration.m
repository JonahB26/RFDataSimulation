% This script is responsible for final acquisiton of data. It will call the
% GenerateMLData function on the tumor/bmode file paths, and return the
% output that I need for my ML project.
% Created by Jonah Boutin on 12/18/2024.
clc,clear,close all

output_path = "C:\Users\1bout\Documents\ElastMLData\";%Path to results folder

tumor_masks_folder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\ClinicalTumorBoundaries-Large';
tumor_files = dir(fullfile(tumor_masks_folder,"*.mat"));
h = waitbar(0,'Starting...');
for i = 1:length(tumor_files)
    waitbar(i/length(tumor_files),h,sprintf('Processing %d of %d',i,length(tumor_files)));
    ImagePath = fullfile(tumor_masks_folder,tumor_files(i).name);
    load(ImagePath)
    
    tumor_mask = tumor_area;
    tumor_mask = imresize(double(tumor_mask),[256,256]);
    tumor_mask = tumor_mask > 0.5;
    f
    ligament_thickness = 0.4475;
    ligament_stiffness = 150000;

    if i < 0.85*length(tumor_files)
       label = 'malignant';
       tumor_median = 13000;
    else
       label = 'benign';
       tumor_median = 4000;
    end
    
    [simresult,cooper_mask,shape,lateral_boundary_corners,axial_boundary_corners,YM_final] = FiniteElementAnalysisBatch( ...
    tumor_mask,label,tumor_median,ligament_thickness,ligament_stiffness,false);

    [Frame1,Frame2,transducer_num] = FieldIIBatch(simresult,cooper_mask,tumor_mask,label);

    reconstruction_result = GenerateElastographyImage(Frame1,Frame2,false);

    output = struct();

    output.images.tumor_mask = tumor_mask;
    output.images.cooper_mask = cooper_mask;
    output.images.ym_ground_truth = YM_final;
    output.images.elastography_image = reconstruction_result;

    output.finite_element_info.axial_boundary_corners = axial_boundary_corners;
    output.finite_element_info.lateral_boundary_corners = lateral_boundary_corners;
    output.finite_element_info.output_struct = simresult;

    output.tumor_info.label = label;
    output.tumor_info.tumor_median = tumor_median;

    output.field_ii_info.transducer_num = transducer_num;
    output.field_ii_info.frame_one = Frame1;
    output.field_ii_info.frame_two = Frame2;

    name = strcat("ElastData_",num2str(i));
    filename = output_path+name+".mat";
    save(char(filename), "output");
    fprintf('Finished number %d',i);
end
% 
% for i = 2%:length(BusiBenignImageFiles)
%     % disp(i);
%    %  if skipCounter > 0
%    %     skipCounter = skipCounter - 1;
%    %     continue
%    %  end
%    % 
%     ImagePath = fullfile(BusiBenignImageFolder,BusiBenignImageFiles(i).name);
%     % [~, filename, ~] = fileparts(BusiBenignImageFiles(i).name);
%    % 
%    %  [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(filename,BusiBenignTumorFiles,BusiBenignTumorMasksFolder);
%    % 
%    %  % Make tumor mask
%    %  tumor_mask = logical(imread(TumorMaskPath));
%    %     %Check to see if there is a second tumor mask for this file.
%    % if tumorFileLocation + 1 <= length(BusiBenignTumorFiles)
%    %      %Check to see if there is a second tumor mask for this file.
%    %      nextFile = BusiBenignTumorFiles(tumorFileLocation + 1).name;
%    %      [startindB,endindB] = regexp(nextFile,'\d*');
%    %      numnextTumorFile = nextFile(startindB:endindB);
%    %      if strcmp(numImageFile,numnextTumorFile)
%    %          TumorMaskPath1 = fullfile(BusiBenignTumorMasksFolder,nextFile);
%    %          tumor_mask1 = imread(TumorMaskPath1);
%    %          tumor_mask(:,:,2) = logical(tumor_mask1(1:size(tumor_mask,1),1:size(tumor_mask,2)));
%    %          disp(TumorMaskPath1)
%    %          skipCounter = 1;
%    %      end
%    % end
%    % 
%    % %Check to see if there is a third tumor mask for this file.
%    % if tumorFileLocation + 2 <= length(BusiBenignTumorFiles)
%    %      nextFile2 = BusiBenignTumorFiles(tumorFileLocation + 2).name;
%    %      [startindB,endindB] = regexp(nextFile2,'\d*');
%    %      numnextnextTumorFile = nextFile2(startindB:endindB);
%    %       disp(numnextnextTumorFile)
%    %      if strcmp(numTumorFile,numnextnextTumorFile)
%    %          TumorMaskPath2 = fullfile(BusiBenignTumorMasksFolder,nextFile2);
%    %          tumor_mask2 = imread(TumorMaskPath2);
%    %          tumor_mask(:,:,3) = logical(tumor_mask2(1:size(tumor_mask,1),1:size(tumor_mask,2)));
%    %          disp(TumorMaskPath2)
%    %          skipCounter = 2;
%    %      end
%    % end
%    % 
%    % %Build final tumor mask
%    % tumor_mask_final = zeros(size(tumor_mask(:,:,1)));
%    % for j = 1:size(tumor_mask,3)
%    %      tumor_mask_final = tumor_mask_final + tumor_mask(:,:,j);
%    % end
%    % 
%    % %Call big ass function NEEDS TUMOR MIDDLE NUM AND LABEL FOR ML
%    % tumor_mask_final = logical(tumor_mask_final);
%    tumor = load(ImagePath);
%    tumor_mask_final = tumor.tumor_area;
%    % tumor_mask_final = bwareaopen(imclose(tumor_mask_final, strel('disk', 5)), 60); % Clean mask
%    ImagePath = tumor_mask_final;
% 
%    if i < 0.85*length(BusiBenignImageFiles)
%        label = 'malignant';
%        tumor_median = 13000;
%    else
%        label = 'benign';
%        tumor_median = 4000;
%    end
% 
%    result = GenerateMLData(tumor_mask_final,ImagePath,tumor_median,label);
% 
%    % Need to save to a folder.
%    % file_hex = DataHash(result, 'array','hex');
% 
%     % filename = strcat(output_path, file_hex,".mat");
%     % save(filename, "result")
%     name = strcat("ElastData_",num2str(i));
%     filename = output_path+name+".mat";
%     save(char(filename), "result");
%     fprintf('Finished number %d',i);
% end
% 
% % clearvars -except output_path; %Clean up
% % % Then BUSI malignant
% % % Run code on BUSIbenign files.
% % BusiMalignantImageFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\malignantbusi';
% % BusiMalignantImageFiles = dir(fullfile(BusiMalignantImageFolder,"*.png"));
% % BusiMalignantTumorMasksFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\TumorMasksOrganized\BUSIMalignant';
% % BusiMalignantTumorFiles = dir(fullfile(BusiMalignantTumorMasksFolder,"*.png"));
% % skipCounter = 0;
% % for i = 1%:length(BusiMalignantImageFiles)
% %     if skipCounter > 0
% %        skipCounter = skipCounter - 1;
% %        continue
% %     end
% % 
% %     ImagePath = fullfile(BusiMalignantImageFolder,BusiMalignantImageFiles(i).name);
% %     [~, filename, ~] = fileparts(BusiMalignantImageFiles(i).name);
% % 
% %     [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(filename,BusiMalignantTumorFiles,BusiMalignantTumorMasksFolder);
% % 
% %     % Make tumor mask
% %     tumor_mask = logical(imread(TumorMaskPath));
% %        %Check to see if there is a second tumor mask for this file.
% %    if tumorFileLocation + 1 <= length(BusiMalignantTumorFiles)
% %         %Check to see if there is a second tumor mask for this file.
% %         nextFile = BusiMalignantTumorFiles(tumorFileLocation + 1).name;
% %         [startindB,endindB] = regexp(nextFile,'\d*');
% %         numnextTumorFile = nextFile(startindB:endindB);
% %         if strcmp(numImageFile,numnextTumorFile)
% %             TumorMaskPath1 = fullfile(BusiMalignantTumorMasksFolder,nextFile);
% %             tumor_mask1 = imread(TumorMaskPath1);
% %             tumor_mask(:,:,2) = logical(tumor_mask1(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath1)
% %             skipCounter = 1;
% %         end
% %    end
% % 
% %    %Check to see if there is a third tumor mask for this file.
% %    if tumorFileLocation + 2 <= length(BusiMalignantTumorFiles)
% %         nextFile2 = BusiMalignantTumorFiles(tumorFileLocation + 2).name;
% %         [startindB,endindB] = regexp(nextFile2,'\d*');
% %         numnextnextTumorFile = nextFile2(startindB:endindB);
% %          disp(numnextnextTumorFile)
% %         if strcmp(numTumorFile,numnextnextTumorFile)
% %             TumorMaskPath2 = fullfile(BusiMalignantTumorMasksFolder,nextFile2);
% %             tumor_mask2 = imread(TumorMaskPath2);
% %             tumor_mask(:,:,3) = logical(tumor_mask2(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath2)
% %             skipCounter = 2;
% %         end
% %    end
% % 
% %    %Build final tumor mask
% %    tumor_mask = zeros(size(tumor_mask(:,:,1)));
% %    for j = 1:size(tumor_mask,3)
% %         tumor_mask = tumor_mask + tumor_mask(:,:,j);
% %    end
% % 
% %    %Call big ass function NEEDS TUMOR MIDDLE NUM AND LABEL FOR ML
% %    tumor_mask = logical(tumor_mask);
% %    result = GenerateMLData(tumor_mask,ImagePath,15000,'malignant');
% % 
% %    % Need to save to a folder.
% %    file_hex = DataHash(result, 'array','hex');
% % 
% %     filename = strcat(output_path, file_hex,".mat");
% %     save(filename, "result")
% % 
% %     fprintf('Finished Busi malignant number %d',i);
% % 
% % 
% % end
% % 
% % clearvars -except output_path; %Clean up
% % % First BUSBRA
% % % Run code on BUSBRA files.
% % BUSBRAImageFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\ImagesBUSBRA';
% % BUSBRAImageFiles = dir(fullfile(BUSBRAImageFolder,"*.png"));
% % BUSBRATumorMasksFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\TumorMasksOrganized\BUSBRA';
% % BUSBRATumorFiles = dir(fullfile(BUSBRATumorMasksFolder,"*.png"));
% % skipCounter = 0;
% % for i = 1%:length(BUSBRAImageFiles)
% %     if skipCounter > 0
% %        skipCounter = skipCounter - 1;
% %        continue
% %     end
% % 
% %     ImagePath = fullfile(BUSBRAImageFolder,BUSBRAImageFiles(i).name);
% %     [~, filename, ~] = fileparts(BUSBRAImageFiles(i).name);
% % 
% %     [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(filename,BUSBRATumorFiles,BUSBRATumorMasksFolder);
% % 
% %     % Make tumor mask
% %     tumor_mask = logical(imread(TumorMaskPath));
% %        %Check to see if there is a second tumor mask for this file.
% %    if tumorFileLocation + 1 <= length(BUSBRATumorFiles)
% %         %Check to see if there is a second tumor mask for this file.
% %         nextFile = BUSBRATumorFiles(tumorFileLocation + 1).name;
% %         [startindB,endindB] = regexp(nextFile,'\d*');
% %         numnextTumorFile = nextFile(startindB:endindB);
% %         if strcmp(numImageFile,numnextTumorFile)
% %             TumorMaskPath1 = fullfile(BUSBRATumorMasksFolder,nextFile);
% %             tumor_mask1 = imread(TumorMaskPath1);
% %             tumor_mask(:,:,2) = logical(tumor_mask1(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath1)
% %             skipCounter = 1;
% %         end
% %    end
% % 
% %    %Check to see if there is a third tumor mask for this file.
% %    if tumorFileLocation + 2 <= length(BUSBRATumorFiles)
% %         nextFile2 = BUSBRATumorFiles(tumorFileLocation + 2).name;
% %         [startindB,endindB] = regexp(nextFile2,'\d*');
% %         numnextnextTumorFile = nextFile2(startindB:endindB);
% %          disp(numnextnextTumorFile)
% %         if strcmp(numTumorFile,numnextnextTumorFile)
% %             TumorMaskPath2 = fullfile(BUSBRATumorMasksFolder,nextFile2);
% %             tumor_mask2 = imread(TumorMaskPath2);
% %             tumor_mask(:,:,3) = logical(tumor_mask2(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath2)
% %             skipCounter = 2;
% %         end
% %    end
% % 
% %    %Build final tumor mask
% %    tumor_mask = zeros(size(tumor_mask(:,:,1)));
% %    for j = 1:size(tumor_mask,3)
% %         tumor_mask = tumor_mask + tumor_mask(:,:,j);
% %    end
% % 
% %    %Call big ass function NEEDS TUMOR MIDDLE NUM AND LABEL FOR ML
% %    tumor_mask = logical(tumor_mask);
% %    result = GenerateMLData(tumor_mask,ImagePath,15000,'malignant');
% % 
% %    % Need to save to a folder.
% %    file_hex = DataHash(result, 'array','hex');
% % 
% %     filename = strcat(output_path, file_hex,".mat");
% %     save(filename, "result")
% % 
% %     fprintf('Finished BUSBRA number %d',i);
% % 
% % 
% % end
% % 
% % clearvars -except output_path; %Clean up
% % % Then BrEAST
% % % Run code on BrEAST files.BusiBenign
% % BrEASTImageFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\ImagesBrEAST';
% % BrEASTImageFiles = dir(fullfile(BrEASTImageFolder,"*.png"));
% % BrEASTTumorMasksFolder = 'C:\Users\1bout\OneDrive\Documents\JonahMEScFiles\data for ML generation\TumorMasksOrganized\BrEAST';
% % BrEASTTumorFiles = dir(fullfile(BrEASTTumorMasksFolder,"*.png"));
% % skipCounter = 0;
% % for i = 1%:length(BrEASTImageFiles)
% %     if skipCounter > 0
% %        skipCounter = skipCounter - 1;
% %        continue
% %     end
% % 
% %     ImagePath = fullfile(BrEASTImageFolder,BrEASTImageFiles(i).name);
% %     [~, filename, ~] = fileparts(BrEASTImageFiles(i).name);
% % 
% %     [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(filename,BrEASTTumorFiles,BrEASTTumorMasksFolder);
% % 
% %     % Make tumor mask
% %     tumor_mask = logical(imread(TumorMaskPath));
% %        %Check to see if there is a second tumor mask for this file.
% %    if tumorFileLocation + 1 <= length(BrEASTTumorFiles)
% %         %Check to see if there is a second tumor mask for this file.
% %         nextFile = BrEASTTumorFiles(tumorFileLocation + 1).name;
% %         [startindB,endindB] = regexp(nextFile,'\d*');
% %         numnextTumorFile = nextFile(startindB:endindB);
% %         if strcmp(numImageFile,numnextTumorFile)
% %             TumorMaskPath1 = fullfile(BrEASTTumorMasksFolder,nextFile);
% %             tumor_mask1 = imread(TumorMaskPath1);
% %             tumor_mask(:,:,2) = logical(tumor_mask1(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath1)
% %             skipCounter = 1;
% %         end
% %    end
% % 
% %    %Check to see if there is a third tumor mask for this file.
% %    if tumorFileLocation + 2 <= length(BrEASTTumorFiles)
% %         nextFile2 = BrEASTTumorFiles(tumorFileLocation + 2).name;
% %         [startindB,endindB] = regexp(nextFile2,'\d*');
% %         numnextnextTumorFile = nextFile2(startindB:endindB);
% %          disp(numnextnextTumorFile)
% %         if strcmp(numTumorFile,numnextnextTumorFile)
% %             TumorMaskPath2 = fullfile(BrEASTTumorMasksFolder,nextFile2);
% %             tumor_mask2 = imread(TumorMaskPath2);
% %             tumor_mask(:,:,3) = logical(tumor_mask2(1:size(tumor_mask,1),1:size(tumor_mask,2)));
% %             disp(TumorMaskPath2)
% %             skipCounter = 2;
% %         end
% %    end
% % 
% %    %Build final tumor mask
% %    tumor_mask = zeros(size(tumor_mask(:,:,1)));
% %    for j = 1:size(tumor_mask,3)
% %         tumor_mask = tumor_mask + tumor_mask(:,:,j);
% %    end
% % 
% %    %Call big ass function NEEDS TUMOR MIDDLE NUM AND LABEL
% %    tumor_mask = logical(tumor_mask);
% %    result = GenerateMLData(tumor_mask,ImagePath,5000,'benign');
% % 
% %    % Need to save to a folder.
% %    file_hex = DataHash(result, 'array','hex');
% % 
% %     filename = strcat(output_path, file_hex,".mat");
% %     save(filename, "result")
% % 
% %     fprintf('Finished BrEAST number %d',i);
% % 
% % 
% % end
