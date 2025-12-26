%% This section organizes the available displacement dataset into a 3D array to be worked with for PCA (third dimension is just numfiles)
% This script performs PCA on available displacement: Add info about Niusha

data_directory = uigetdir;


files = dir(fullfile(data_directory,'*.mat'));

amountOfPoints = 200;%Amount of points to be used from each boundary (top,bottom,right,left).
displacementDataSet = zeros(11,8*amountOfPoints);

tic

i = 1;

j = 0;

rowCounter = 1;

while i <= size(files,1)   
    

    latfile = files(i);
    data1 = load(fullfile(latfile.folder, latfile.name));
    disp(latfile.name)

    fields = fieldnames(data1);

    lateraldisplacements = data1.(fields{1});

    leftBoundaryConditionX = lateraldisplacements(:,3);%Grab the third column of the lateral displacements
    if size(leftBoundaryConditionX,1) ~= amountOfPoints
        x = linspace(0,length(leftBoundaryConditionX),length(leftBoundaryConditionX));
        newx = linspace(max(x), min(x), amountOfPoints);
        leftBoundaryConditionX = interp1(x,leftBoundaryConditionX,newx,"spline");
    end
    displacementDataSet(rowCounter,1:amountOfPoints) = leftBoundaryConditionX;

    rightBoundaryConditionX = lateraldisplacements(:,end-2); %Grab the third last column of the lateral displacements
    if size(rightBoundaryConditionX,1) ~= amountOfPoints
        x = linspace(0,length(rightBoundaryConditionX),length(rightBoundaryConditionX));
        newx = linspace(max(x), min(x), amountOfPoints);
        rightBoundaryConditionX = interp1(x,rightBoundaryConditionX,newx,"spline");
    end
    displacementDataSet(rowCounter,(amountOfPoints+1):(2*amountOfPoints)) = rightBoundaryConditionX;


    bottomBoundaryConditionX = lateraldisplacements(end-2,:); %Grab the third last row of lateral displacements
    if size(bottomBoundaryConditionX,1) ~= amountOfPoints
        x = linspace(0,length(bottomBoundaryConditionX),length(bottomBoundaryConditionX));
        newx = linspace(max(x), min(x), amountOfPoints);
        bottomBoundaryConditionX = interp1(x, bottomBoundaryConditionX,newx,"spline");
    end
    displacementDataSet(rowCounter,(2*amountOfPoints+1):(3*amountOfPoints)) = bottomBoundaryConditionX;


    topBoundaryConditionX = lateraldisplacements(3,:); %Grab the third row of lateral displacements
    if size(topBoundaryConditionX,1) ~= amountOfPoints
        x = linspace(0,length(topBoundaryConditionX),length(topBoundaryConditionX));
        newx = linspace(max(x), min(x), amountOfPoints);
        topBoundaryConditionX = interp1(x, topBoundaryConditionX,newx,"spline");
    end
    displacementDataSet(rowCounter,(3*amountOfPoints+1):(4*amountOfPoints)) = topBoundaryConditionX;




    axfile = files(i+1);
    data2 = load(fullfile(axfile.folder, axfile.name));
    disp(axfile.name)

    fields = fieldnames(data2);

    axialdisplacements = data2.(fields{1});


    leftBoundaryConditionY = axialdisplacements(:,3); %Grab the third column of the axial displacements
    if size(leftBoundaryConditionY,1) ~= amountOfPoints
        x = linspace(0,length(leftBoundaryConditionY),length(leftBoundaryConditionY));
        newx = linspace(max(x), min(x), amountOfPoints);
        leftBoundaryConditionY = interp1(x,leftBoundaryConditionY,newx,"spline");
    end
    displacementDataSet(rowCounter,(4*amountOfPoints+1):(5*amountOfPoints)) = leftBoundaryConditionY;


    rightBoundaryConditionY = axialdisplacements(:,end-2); %Grab the third last column of the axial displacements
    if size(rightBoundaryConditionY,1) ~= amountOfPoints
        x = linspace(0,length(rightBoundaryConditionY),length(rightBoundaryConditionY));
        newx = linspace(max(x), min(x), amountOfPoints);
        rightBoundaryConditionY = interp1(x,rightBoundaryConditionY,newx,"spline");
    end
    displacementDataSet(rowCounter,(5*amountOfPoints+1):(6*amountOfPoints)) = rightBoundaryConditionY;


    bottomBoundaryConditionY = axialdisplacements(end-2,:); %Grab the third last row of axial displacements
    if size(bottomBoundaryConditionY,1) ~= amountOfPoints
        x = linspace(0,length(bottomBoundaryConditionY),length(bottomBoundaryConditionY));
        newx = linspace(max(x), min(x), amountOfPoints);
        bottomBoundaryConditionY = interp1(x, bottomBoundaryConditionY,newx,"spline");
    end
    displacementDataSet(rowCounter,(6*amountOfPoints+1):(7*amountOfPoints)) = bottomBoundaryConditionY;


    topBoundaryConditionY =  axialdisplacements(3,:); %Grab the third row of axial displacements
    if size(topBoundaryConditionY,1) ~= amountOfPoints
        x = linspace(0,length(topBoundaryConditionY),length(topBoundaryConditionY));
        newx = linspace(max(x), min(x), amountOfPoints);
        topBoundaryConditionY = interp1(x, topBoundaryConditionY,newx,"spline");  
    end
    displacementDataSet(rowCounter,(7*amountOfPoints+1):(8*amountOfPoints)) = topBoundaryConditionY;


    i = i + 2;%Get to the next lateral file.

    rowCounter = rowCounter + 1;%Update to next row.

    sprintf('File number %d has been processed.',i)

end

save('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/ClinicalBoundaryConditionsBeforePCA.mat',"displacementDataSet")

toc
    %% This section of code runs the PCA on the data and expands the dataset
clc

[TB_PCA_eigenvectors, TB_PCA_score, TB_PCA_latent, tsquared, explained, mu] = pca(displacementDataSet,'NumComponents',20);

PCA_model.database = TB_PCA_score;
PCA_model.eigenvectors = TB_PCA_eigenvectors;
PCA_model.mu = mu;

save(    '/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/PCA_model.mat',"PCA_model")

semilogx(cumsum(explained));
    
numClinicalData = size(displacementDataSet,1);%Number of clinical data points 
numAddedPoints = 1822 - numClinicalData;%Number of data points to be added, 1822 is total size
added = zeros(numAddedPoints,8*amountOfPoints);
displacementDataSet(numClinicalData+1:numClinicalData+numAddedPoints,:) = added;
weights = linspace(0,1,numAddedPoints);

for i = 1:length(weights)

    selectedData = datasample(TB_PCA_score,2,'Replace',false);

    interpolated = weights(i).*selectedData(1,:) + (1 - weights(i).*selectedData(2,:));
        
    newDataPoint = interpolated*TB_PCA_eigenvectors' + mu;

    displacementDataSet(numClinicalData+i,:) = newDataPoint;

    sprintf('Displacement addition number %d has been developed',i)
end
clinicalBoundaryConditionsAfterPCA = displacementDataSet;

save('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/ClinicalBoundaryConditionsAfterPCA.mat','displacementDataSet')
