rawFolder = "D:\new\BUSIMalignantPostCooperRawFilesUpdated";

rawFiles = dir(fullfile(rawFolder,'*.png'));

matFolder = "D:\new\BUSIMalignantPostCooperMatFilesUpdated";

matFiles = dir(fullfile(matFolder,'*.mat'));

for i = 1:length(matFiles)
    
    [~,matFileName,~] = fileparts(matFiles(i).name);

    fileExists = false;

    for j = 1:length(rawFiles)

        [~,rawFileName,~] = fileparts(rawFiles(j).name);

        rawFileName = strrep(rawFileName,'_raw','');

        if strcmp(matFileName,rawFileName)

            fileExists = true;
            break
        end
    end

    matFilePath = fullfile(matFolder,matFiles(i).name);

    if ~fileExists
        delete(matFilePath)
    else
        continue
    end
    
    disp(['Finished file',num2str(i)])

end


        