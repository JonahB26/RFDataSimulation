function [imfixed] = RemoveAnnotations(USimagePath)
%This function takes the filepath to an ultrasound image with annotations,
%applies a central operator for edge-detection, and proceeds to reassign
%pixel values within the annotation based on neighbouring pixels, and then
%perform image smoothening and sharpening before displaying it.

tic; %Initialize elapsed time counter

%Load in image based on the inputted image path
k = imread(USimagePath);

%If by chance it is a color image, convert to grayscale
if size(k,3) == 3
    k = rgb2gray(k);
end

%Convert to datatype double
k1 = double(k);

%edges = edge(k1,'canny');


%Use a central operator for edge detection
forward_mask = [1/2 0 -1/2];
Kx = conv2(k1,forward_mask,'same');

%Use the edge function to further grab the edges, and then binarize that
%image
BW = edge(Kx,'roberts');
BW = double(BW);
BW = imbinarize(BW);

% %Just for testing
% BW = edge(k1,'canny');
% BW = uint8(BW);
% BW = imbinarize(BW);

%Create structuring element to use imdilate and expand the boundaries
se = strel('disk', 4);
dilated = imdilate(BW,se);
%dilated = imclose(BW,se);
% dilatedEdges = imdilate(edges,se);
% 
% filledRegions = imfill(dilatedEdges,'holes');
% 
% kRestored = regionfill(k1,filledRegions);
% 
% kRestored = imgaussfilt(kRestored,1);
% 
% kRestored = imsharpen(kRestored,'Amount',1.3);
% 
% figure
% imshow(kRestored)

%Fill in the gaps in the edges
filled = imfill(dilated,'holes');

%Gather the locations of every pixel in one of the annotations
[row, col] = find(filled);
locations = [row, col];
annotationmask = zeros(size(k,1),size(k,2));
annotationmask(sub2ind(size(annotationmask), locations(:,1),locations(:,2))) = true;
annotationmask = logical(annotationmask);
% annotationmask = k(locations(1),locations(2));

% mask = filled;
% k = regionfill(k,mask);
% 
% k = imadjust(k);

neighbourhood = zeros(size(k,1),size(k,2));
% range = zeros(11);
%For-loop for setting new pixel values within the annotations
for i = 1:size(locations,1)
    neighbourhood(:) = 0;

    %Define xy-location of the pixel of interest in the iteration
    xlocation = locations(i,1);
    ylocation = locations(i,2);
    % 
    % neighborhoodSize = 3;
    % 
    % xStart = max(1,xlocation - floor(neighborhoodSize / 2));
    % xEnd = min(size(k,1),xlocation + floor(neighborhoodSize / 2));
    % 
    % ystart = max(1,ylocation - floor(neighborhoodSize / 2));
    % yEnd = min(size(k,2),ylocation + floor(neighborhoodSize / 2));
    % 
    % localNeighborhood = k(xStart:xEnd,ystart:yEnd);
    % 
    % newvalue = mean(localNeighborhood(:));
    % 
    % k(xlocation,ylocation) = newvalue;

    %Determine the location of a new pixel value to be assigned within that pixel of the
    %annotation
    % newvaluexlocation = xlocation + 10;
    % newvalueylocation = ylocation -20;
    
    [rows,cols] = size(k);
    % disp(['x: ',num2str(xlocation)])
    if (xlocation-5) <= 0 
        xlocation = 6;
        % disp(['new x: ',num2str(xlocation)])
    end
    % disp(['x: ',num2str(xlocation)])
    if (xlocation+5) >= rows
        xlocation = rows - 5;
        % disp(['new x: ',num2str(xlocation)])
    end
    % disp(['y: ',num2str(ylocation)])
    if (ylocation-5) <= 0
        ylocation = 6;
        % disp(['new y: ',num2str(ylocation)])
    end
    % disp(['y: ',num2str(ylocation)])
    if (ylocation+5) >= cols
        ylocation = cols - 5;
        % disp(['new y: ',num2str(ylocation)])
    end
    % [rows,cols] = size(k);
    % disp(['Xlocation: ',num2str(xlocation),' Ylocation: ',num2str(ylocation),' Sizex: ',num2str(rows),' Sizey: ',num2str(cols)])
    
    ROI = k(xlocation-5:xlocation+5,ylocation-5:ylocation+5);
    % mean(ROI(:))
    neighbourhood(xlocation-5:xlocation+5,ylocation-5:ylocation+5) = ROI;
    neighbourhoodmask = neighbourhood > 0;
    % ROI = k(xlocation-5:xlocation+5,ylocation-5:ylocation+5)
    % neighbourhoodmask = neighbourhood > 0;

    % disp(['hood mask size: ',num2str(size(neighbourhoodmask)),' ann mask size: ',size(annotationmask)])
    neighbourhoodmask(annotationmask) = 0;
    

    %Set continue condition to move to the next pixel if the location
    %exceeds the boundaries
    % if xlocation- >= size(filled,1) || newvalueylocation >= size(filled,2) || newvalueylocation <= 0
    %     continue
    % end

    %Determine the new pixel value assigned to the pixel in the annotation
    % newvalue = k(newvaluexlocation,newvalueylocation);
    values = k(neighbourhoodmask);
    newvalue = mean(values(:),'omitnan');
    % disp(['mean is ',num2str(newvalue)])

    %Assign that pixel value
    k(xlocation,ylocation) = newvalue;

end

%Apply a gaussian filter to smoothen the image, then immediately sharpen
%the image
k = imgaussfilt(k,1);
k = imsharpen(k,'Amount',1.3);
imfixed = k;
%Display the image
% if visualize == 'true'
% figure
% imshow(k)
% close all
% end

toc %Finalize the elapsed time counter
end