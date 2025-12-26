%This script will assign mechanical properties and use Matthew Caius's
%Finite Element code to calculate the displacements

%There are 211 malignant BUSI masks, 454 benign BUSI masks, 1863 BUSBRA
%masks, 252 BReAST masks. The masks are organized, benign BUSI, BReAST,
%malignant BUSI, BUSBRA.
tic

addpath "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/Outputs/matFilesOrganizedPost/"
addpath "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/TumorMasksOrganized/"
addpath "/home/deeplearningtower/Documents/MATLAB/JonahCode/MC-Elastosynth/"
addpath "/home/deeplearningtower/Documents/MATLAB/JonahCode/MC-Elastosynth/FEM Interface Windows/"
%When running this from lab desktop manually add MC-Elastosynth to path,
%and comment lines 11&12.
%Just load one image as example

% ImageFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MatFilesOrganized";
ImageFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/BreastCancerDiagnosis-ML/TestFilesMat";
%ImageFolder = uigetdir; %Comment above line and use this when at home
ImageFiles = dir(fullfile(ImageFolder,"*.mat"));

% TumorMasksFolder = "D:\TumorMasksOrganized\BrEAST\";
% TumorMasksFolder = "D:\TumorMasksOrganized\BUSBRA;
% TumorMasksFolder = "D:\TumorMasksOrganized\BUSIBenign";
% TumorMasksFolder = "D:\TumorMasksOrganized\BUSIMalignant";
% TumorFiles = dir(fullfile(TumorMasksFolder,"*.png"));


%for i in range of 1:230 use the busi benign, for 231:401 use BrEAST,
%for 402:1186 use BUSBRA, for 1187:length(ImageFiles) use busi malignant.

%Run code on BUSIbenign files.
TumorMasksFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/TumorMasksOrganized/BUSIBenign";
TumorFiles = dir(fullfile(TumorMasksFolder,"*.png"));
skipCounter = 0;
for i = 10%:337
%    % if exist(fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']),'file')
%    %     disp('File exists. Moving on')
%    %     continue
%    % end
%    outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']);
%    %Skip necessary files depending on if there was more tumor masks.
   if skipCounter > 0
       skipCounter = skipCounter - 1;
       continue
   end

   [~, filename, ~] = fileparts(ImageFiles(i).name);
   outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/BreastCancerDiagnosis-ML/TestFilesMat",[filename,'_075STIFFNESS_Output.mat']);


   disp(filename)

   ImagePath = fullfile(ImageFolder,ImageFiles(i).name);


   CurrentImageFile = filename;

   [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(CurrentImageFile,TumorFiles,TumorMasksFolder);
   disp(TumorMaskPath)

   %Check to see if there is a second tumor mask for this file.
   if tumorFileLocation + 1 <= length(TumorFiles)
        %Check to see if there is a second tumor mask for this file.
        nextFile = TumorFiles(tumorFileLocation + 1).name;
        [startindB,endindB] = regexp(nextFile,'\d*');
        numnextTumorFile = nextFile(startindB:endindB);
        if strcmp(numImageFile,numnextTumorFile)
            TumorMaskPath1 = fullfile(TumorMasksFolder,nextFile);
            disp(TumorMaskPath1)
            skipCounter = 1;
        end
   end

   %Check to see if there is a third tumor mask for this file.
   if tumorFileLocation + 2 <= length(TumorFiles)
        nextFile2 = TumorFiles(tumorFileLocation + 2).name;
        [startindB,endindB] = regexp(nextFile2,'\d*');
        numnextnextTumorFile = nextFile2(startindB:endindB);
         disp(numnextnextTumorFile)
        if strcmp(numTumorFile,numnextnextTumorFile)
            TumorMaskPath2 = fullfile(TumorMasksFolder,nextFile2);
            disp(TumorMaskPath2)
            skipCounter = 2;
        end
   end

   %Provide correct number of inputs to function using skipCounter to
   %determine how many tumor masks there are.
   if skipCounter == 1
        disp('Two masks')
       [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1);
   elseif skipCounter == 2
        disp('Three masks')
       [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1,TumorMaskPath2);
   else
       disp('One mask')
       [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath);
   end


   Output.YM_Image = YM_Image;
   Output.minTumorYM = minTumorYM;
   Output.maxTumorYM = maxTumorYM;
   Output.cooperImageData = cooperImageData;
   Output.cooperImageData.TumorMask = TumorMask;

   save(outPath,'Output');

   sprintf("File number %d has been processed",i)
end

%Run code on BrEAST files.
% TumorMasksFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/TumorMasksOrganized/BrEAST";
% TumorFiles = dir(fullfile(TumorMasksFolder,"*.png"));
% for i = 1502:1680
%   % if exist(fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']),'file')
%   %      disp('File exists. Moving on')
%   %      continue
%   %  end
%    outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']);
%      %Skip necessary files depending on if there was more tumor masks.
%    if skipCounter > 0
%        skipCounter = skipCounter - 1;
%        continue
%    end
% 
%    [~, filename, ~] = fileparts(ImageFiles(i).name);
% 
%    disp(filename)
% 
%    ImagePath = fullfile(ImageFolder,ImageFiles(i).name);
% 
% 
%    CurrentImageFile = filename;
% 
%    [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(CurrentImageFile,TumorFiles,TumorMasksFolder);
%    disp(TumorMaskPath)
% 
%    %Check to see if there is a second tumor mask for this file.
%    if tumorFileLocation + 1 <= length(TumorFiles)
%         %Check to see if there is a second tumor mask for this file.
%         nextFile = TumorFiles(tumorFileLocation + 1).name;
%         [startindB,endindB] = regexp(nextFile,'\d*');
%         numnextTumorFile = nextFile(startindB:endindB);
%         if strcmp(numImageFile,numnextTumorFile)
%             TumorMaskPath1 = fullfile(TumorMasksFolder,nextFile);
%             disp(TumorMaskPath1)
%             skipCounter = 1;
%         end
%    end
% 
%    %Check to see if there is a third tumor mask for this file.
%    if tumorFileLocation + 2 <= length(TumorFiles)
%         nextFile2 = TumorFiles(tumorFileLocation + 2).name;
%         [startindB,endindB] = regexp(nextFile2,'\d*');
%         numnextnextTumorFile = nextFile2(startindB:endindB);
%          disp(numnextnextTumorFile)
%         if strcmp(numTumorFile,numnextnextTumorFile)
%             TumorMaskPath2 = fullfile(TumorMasksFolder,nextFile2);
%             disp(TumorMaskPath2)
%             skipCounter = 2;
%         end
%    end
%    %Provide correct number of inputs to function using skipCounter to
%    %determine how many tumor masks there are.
%    if skipCounter == 1
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1);
%         disp('Two masks')
%    elseif skipCounter == 2
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1,TumorMaskPath2);
%         disp('Three masks')
%    else
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath);
%        disp('One mask')
%    end
% 
% 
%    Output.YM_Image = YM_Image;
%    Output.minTumorYM = minTumorYM;
%    Output.maxTumorYM = maxTumorYM;
%    Output.Image = Image;
% 
%    save(outPath,'Output');
% 
%    sprintf("File number %d has been processed",i)
% end

% % %Run code on BUSBRA files.
% TumorMasksFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/TumorMasksOrganized/BUSBRA";
% TumorFiles = dir(fullfile(TumorMasksFolder,"*.png"));
% skipCounter = 0;
%  for i = 12%338:1501
%    % if exist(fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']),'file')
%    %     disp('File exists. Moving on')
%    %     continue
%    % end
%    %outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']);
%       %Skip necessary files depending on if there was more tumor masks.
%    if skipCounter > 0
%        skipCounter = skipCounter - 1;
%        continue
%    end
%     outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/BreastCancerDiagnosis-ML/TestFilesMat",[filename,'_Output.mat']);
%    [~, filename, ~] = fileparts(ImageFiles(i).name);
% 
%    disp(filename)
% 
%    ImagePath = fullfile(ImageFolder,ImageFiles(i).name);
% 
% 
%    CurrentImageFile = filename;
% 
%    [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(CurrentImageFile,TumorFiles,TumorMasksFolder);
%    disp(TumorMaskPath)
% 
% 
%    %There are no multiple masks in the BUSBRA dataset 
% 
%    % %Check to see if there is a second tumor mask for this file.
%    % if tumorFileLocation + 1 <= length(TumorFiles)
%    %      %Check to see if there is a second tumor mask for this file.
%    %      nextFile = TumorFiles(tumorFileLocation + 1).name;
%    %      [startindB,endindB] = regexp(nextFile,'\d*');
%    %      numnextTumorFile = nextFile(startindB:endindB);
%    %      if strcmp(numTumorFile,numnextTumorFile)
%    %          TumorMaskPath1 = fullfile(TumorMasksFolder,nextFile);
%    %          disp(TumorMaskPath1)
%    %          skipCounter = 1;
%    %      end
%    % end
%    % 
%    % %Check to see if there is a third tumor mask for this file.
%    % if tumorFileLocation + 2 <= length(TumorFiles)
%    %      nextFile2 = TumorFiles(tumorFileLocation + 2).name;
%    %      [startindB,endindB] = regexp(nextFile2,'\d*');
%    %      numnextnextTumorFile = nextFile2(startindB:endindB);
%    %       disp(numnextnextTumorFile)
%    %      if strcmp(numTumorFile,numnextnextTumorFile)
%    %          TumorMaskPath2 = fullfile(TumorMasksFolder,nextFile2);
%    %          disp(TumorMaskPath2)
%    %          skipCounter = 2;
%    %      end
%    % end
% 
%    %Provide correct number of inputs to function using skipCounter to
%    %determine how many tumor masks there are.
%    if skipCounter == 1
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1);
%         disp('Two masks')
%    elseif skipCounter == 2
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1,TumorMaskPath2);
%         disp('Three masks')
%    else
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath);
%        disp('One mask')
%    end
% 
% 
%    Output.YM_Image = YM_Image;
%    Output.minTumorYM = minTumorYM;
%    Output.maxTumorYM = maxTumorYM;
%    Output.cooperImageData = cooperImageData;
%    Output.cooperImageData.TumorMask = TumorMask;
% 
%    save(outPath,'Output');
% 
%    sprintf("File number %d has been processed",i)
% end

% %Run code on BUSIMalignant files.
% TumorMasksFolder = "/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/TumorMasksOrganized/BUSIMalignant";
% TumorFiles = dir(fullfile(TumorMasksFolder,"*.png"));
% for i = 1678:length(ImageFiles)
%    % if exist(fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']),'file')
%    %     disp('File exists. Moving on')
%    %     continue
%    % end
%    outPath = fullfile("/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs/",[filename,'_Output.mat']);
%       %Skip necessary files depending on if there was more tumor masks.
%    if skipCounter > 0
%        skipCounter = skipCounter - 1;
%        continue
%    end
% 
%    [~, filename, ~] = fileparts(ImageFiles(i).name);
% 
%    disp(filename)
% 
%    ImagePath = fullfile(ImageFolder,ImageFiles(i).name);
% 
% 
%    CurrentImageFile = filename;
% 
%    [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(CurrentImageFile,TumorFiles,TumorMasksFolder);
%    disp(TumorMaskPath)
% 
%    %Check to see if there is a second tumor mask for this file.
%    if tumorFileLocation + 1 <= length(TumorFiles)
%         %Check to see if there is a second tumor mask for this file.
%         nextFile = TumorFiles(tumorFileLocation + 1).name;
%         [startindB,endindB] = regexp(nextFile,'\d*');
%         numnextTumorFile = nextFile(startindB:endindB);
%         if strcmp(numImageFile,numnextTumorFile)
%             TumorMaskPath1 = fullfile(TumorMasksFolder,nextFile);
%             disp(TumorMaskPath1)
%             skipCounter = 1;
%         end
%    end
% 
%    %Check to see if there is a third tumor mask for this file.
%    if tumorFileLocation + 2 <= length(TumorFiles)
%         nextFile2 = TumorFiles(tumorFileLocation + 2).name;
%         [startindB,endindB] = regexp(nextFile2,'\d*');
%         numnextnextTumorFile = nextFile2(startindB:endindB);
%          disp(numnextnextTumorFile)
%         if strcmp(numTumorFile,numnextnextTumorFile)
%             TumorMaskPath2 = fullfile(TumorMasksFolder,nextFile2);
%             disp(TumorMaskPath2)
%             skipCounter = 2;
%         end
%    end
% 
%    %Provide correct number of inputs to function using skipCounter to
%    %determine how many tumor masks there are.
%    if skipCounter == 1
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1);
%         disp('Two masks')
%    elseif skipCounter == 2
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath,TumorMaskPath1,TumorMaskPath2);
%         disp('Three masks')
%    else
% 
%        [YM_Image,minTumorYM,maxTumorYM,Image] = obtainYM_Image(ImagePath,TumorMaskPath);
%        disp('One mask')
%    end
% 
% 
%    Output.YM_Image = YM_Image;
%    Output.minTumorYM = minTumorYM;
%    Output.maxTumorYM = maxTumorYM;
%    Output.Image = Image;
% 
%    save(outPath,'Output');
% 
%    sprintf("File number %d has been processed",i)
% end
% toc
   
%% Run the FEA code and obtain lateral and axial resolutions
% Use this command to rebuild linux FEM library if necessary: clibgen.buildInterface({'FEM Src/CoordinateSystem.cpp', 'FEM Src/CoordinateSystem.h', 'FEM Src/FiniteElementModel.cpp', 'FEM Src/FiniteElementModel.h', 'FEM Src/FiniteElementResult.cpp', 'FEM Src/FiniteElementResult.h', 'FEM Src/Mesh.cpp', 'FEM Src/Mesh.h', 'FEM Src/Elements.cpp', 'FEM Src/Elements.h', 'FEM Src/BoundaryCondition.cpp', 'FEM Src/BoundaryCondition.h', 'FEM Src/Material.cpp', 'FEM Src/Material.h', 'FEM Src/MeshGenerator.cpp', 'FEM Src/MeshGenerator.h', 'FEM Src/FEM_Interface.h', 'FEM Src/FEM_Interface.cpp', 'FEM Src/MatrixGeneration.h', 'FEM Src/MatrixGeneration.cpp'},'Libraries','FEM Interface Linux/libReconstruction_Library.so','InterfaceName','FEM_Interface');
addpath(genpath("MC-Elastosynth")) %Adds all Elastosynth files to path.
tic

OutputFolder = '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/OutputYMs';

OutputFiles = dir(fullfile(OutputFolder, '*.mat'));

displacements = load('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/ClinicalBoundaryConditionsAfterPCA.mat');

for i = 1:2%length(OutputFiles)

    [~, filename, ~] = fileparts(OutputFiles(i).name);
    
    currentFile = filename;
    data = load(fullfile(OutputFolder,filename));
    disp(currentFile)

    axialresolution = size(data.Output.YM_Image,1);
    lateralresolution = size(data.Output.YM_Image,2);
    
    material = Material(data.Output.YM_Image, 0.45);
    analysis_options = FEMOpts("cartesian", axialresolution, lateralresolution, "PLANE_STRESS"); 
    
    boundary_conditions = BoundaryConditions();
    boundary_conditions.top_axial = displacements.displacementDataSet(i,1401:1600);
    boundary_conditions.bottom_axial = displacements.displacementDataSet(i,1201:1400);
    
    boundary_conditions.top_lateral = displacements.displacementDataSet(i,601:800);
    boundary_conditions.bottom_lateral = displacements.displacementDataSet(i,401:600);
    
    boundary_conditions.right_axial = displacements.displacementDataSet(i,1001:1200);
    boundary_conditions.left_axial = displacements.displacementDataSet(i,801:1000);
    
    boundary_conditions.right_lateral = displacements.displacementDataSet(i,201:400);
    boundary_conditions.left_lateral = displacements.displacementDataSet(i,1:200);
    
    
    % Verify and modify the boundary conditions to match the size of the
    % YM_Image.
    boundary_conditions = verifyLengthOfBoundaries(boundary_conditions, ...
        lateralresolution,axialresolution);
    
    % I've made a slight modification in this function so the visualize option
    % only shows me the axial and lateral displacements.
    result = RunFiniteElementAnalysis(analysis_options,material,boundary_conditions,true);

    Output.FEAdisplacements.LateralDisplacements = result.lateral_disp;
    Output.FEAdisplacements.AxialDisplacements = result.axial_disp;
    
    NameSave = fullfile(OutputFolder,OutputFiles(i).name);
    save(NameSave,"-append")
end
toc