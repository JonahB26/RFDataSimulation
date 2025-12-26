% This script is for generating an elastography image, generate raw data and then we'll process it in python,
% but for now we'll save a struct with elast image and rf data.

output_path = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\TestElastOutputs\";%Path to results folder

data_folder = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\test_rf_data";
data_files = dir(fullfile(data_folder,"*.mat"));

for i = 1:length(data_files)
    data = fullfile(data_folder,data_files(i).name);
    data_struct = load(data).result;
    % Just need to check format, load in RF frames, then run the script.
    reconstruction_result = GenerateElastographyImage(data_struct.Frame1,data_struct.Frame2,false);

    result = struct();

    result.Frame1 = data_struct.Frame1;
    result.Frame2 = data_struct.Frame2;
    result.file_name = data;
    result.elastography_image = reconstruction_result;
    out_name = "ElastoMLData_" + num2str(i);
    filename = strcat(output_path, out_name,".mat");
    save(filename, "result")
    fprintf('Finished number %d',i);
end