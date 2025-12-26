% Set the folder containing the .png files
folderPath = "D:\BUSBRAPostCooperRawFiles";  % Replace with your folder path

% Get a list of all .png files in the folder
pngFiles = dir(fullfile(folderPath, '*.png'));

% Instructions for the user
disp('Press the "Delete" key to delete an image. Press any other key to keep it.');

% Loop through each .png file
counter = 0;
for k = 279:length(pngFiles)
    % Load the .png image
    imagePath = fullfile(folderPath, pngFiles(k).name);
    image = imread(imagePath);  % Read the image file
    
    % Display the image
    figure;
    imshow(image);
    title(['Image: ', pngFiles(k).name]);
    
    % Wait for user key press
    keyPressed = waitforbuttonpress;
    key = get(gcf, 'CurrentKey');
    
    % If the delete key is pressed, delete the file
    if strcmp(key, 'delete')
        % Delete the .png file
        delete(imagePath);
        disp(['Deleted: ', pngFiles(k).name]);
    else
        disp(['Kept: ', pngFiles(k).name]);
    end
    
    % Close the figure
    close(gcf);

    counter = counter + 1;

    disp(['File num: ',counter]);

    filesRemaining = length(pngFiles) - k;
    disp(['Files processed: ', num2str(counter)]);
    disp(['Files remaining: ', num2str(filesRemaining)]);

end

disp('Finished processing all images.');

% Last file deleted was bus_504-r. Set k to 945. (Recheck tho it might
% change.