function saveMutualInformationClinicalStatistics(MIfilePath)

    % Load the data
    data = load(MIfilePath);
    mutual_information_values = data.mutual_information_values;

    % Assign dictionary values
    miStats = dictionary;
    miStats('min MI value') = min(mutual_information_values);
    miStats('max MI value') = max(mutual_information_values);
    miStats('mean MI value') = mean(mutual_information_values);
    miStats('std of MI values') = std(mutual_information_values);

    % Save the dictionary to the MutualInformationFolder
    save('C:\Users\1bout\OneDrive\SamaniLab\MESc Files\MESc Code Files - git\Data Generation\MutualInformation\MutualInformationStatisticsClinicalDataOutputs.mat',"miStats")

