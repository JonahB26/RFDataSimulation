function  [cooper_mask,cooper_image] = AddCoopersLigaments(tumor_mask,lateralres,axialres,theta,thickness,convergence_region)

%This function adds Cooper's Ligaments to the phantom that is previously
%created. This process was done by using the rotation matrix, in order to
%plot both sinusoidal functions as well as their rotated counterparts.
%Lateralres is the lateral resolution of the phantom, and theta is the
%angle by which the original sinusoid is rotated, in radians. The output is
%the data produced, x and y being the original sinusoid, and x_r and y_r
%being the rotated sinusoid. This function has now been updated to contain
%more ligaments, to account for smaller images. convergence_region must be
%either 'top-left', 'top-right', 'bottom-left',bottom-right','center'.

hold on

%numligs = 2;%Set the number of ligaments to appear on the phantom
%shifts = [lateralres/3  2*lateralres/3 2.8*lateralres/3];%linspace(0,axialres,numligs);%Create the array for the shifts, 

shifts = linspace(lateralres/5,4.9*lateralres,10);

%these are the vertical locations of each ligament, should randomize later,
%changed t0 /4 instead of /3

%COMMENTED FOR OPIMIZATION
% The following two are to randomize location if you don't want to do convergence region
% shiftx = randi([500 750],1,1);
% shifty = randi([500 800],1,1);
% shiftx = 607;
% shifty = 517;

switch convergence_region
    case 'top-left'
        shiftx = 200;
        shifty = -200;
    case 'top-right'
        shiftx = lateralres + 200;
        shifty = -300;
    case 'bottom-left'
        shiftx = 200;
        shifty = axialres + 200;
    case 'bottom-right'
        shiftx = lateralres + 200;
        shifty = axialres + 200;
    case 'center'
        shiftx = lateralres / 2;
        shifty = axialres / 2;
    otherwise
        error('Invalid convergence region. Choose from: top-left, top-right, bottom-left, bottom-right, center');
end

%COMMENTED FOR OPTIMIZATION
randomsign = randi([-1 1],1,1);
if randomsign == 0
    while randomsign == 0
        randomsign = randi([-1 1],1,1);
    end
end
% randomsign = -1;

for i = 1:length(shifts)
     % length = randi([100,150]);%Randomly select length of ligament 
     % maxstartingpoint = lateralres - length + 1;%Set furthest possible starting point for ligament
     % startingpoint = randi([1,maxstartingpoint]);%Randomly select horizontal starting point of ligament
     % x = startingpoint:(startingpoint + length-1);%Use starting point and length to generate x-values
    x = -500:0.5:2000;
    y = randomsign*110*sin((1/230)*x-15)+shifts(i);%Evaluate function at x-values
    % theta = pi/2;

    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    rotated = R*[x; y];
    x_r = rotated(1,:);
    x_r = x_r + shiftx;% + shifts(i);
    y_r = rotated(2,:);
    y_r = y_r - shifty;

    % x1 = x*cos(rotateangle) - y*sin(rotateangle);
    % y1 = x*sin(rotateangle) + y*cos(rotateangle);
    
% <<<<<<< HEAD:AddCoopersLigaments.m
%     plot(x,y,Color=[1 1 1],LineWidth=1.5);%Plot the ligaments over the phantom
%     plot(x_r,y_r,Color=[1 1 1],LineWidth=1.5);%Plot the ligaments over the phantom
% =======
    % Modified these so that user can input ligament thickness
    plot(x,y,Color='r',LineWidth=thickness);%Plot the ligaments over the phantom
    plot(x_r,y_r,Color='r',LineWidth=thickness);%Plot the ligaments over the phantom
% >>>>>>> bc846e5e2df480e9c9ebfdf1415ec87892d556c4:Data Generation/AddCoopersLigaments.m
end

exportgraphics(gcf,'intermediate_cooper_figure.png', ...
    'Resolution',300, 'BackgroundColor','white', 'ContentType','image');

cooper_image = imread('intermediate_cooper_figure.png');
cooper_image = imresize(cooper_image,[axialres lateralres],'nearest'); % avoid blur

% Convert the image to HSV color space
hsv_image = rgb2hsv(cooper_image);

% Extract the HSV channels
hue_channel = hsv_image(:, :, 1);    % Hue
saturation_channel = hsv_image(:, :, 2); % Saturation
value_channel = hsv_image(:, :, 3); % Value

% Threshold the red color based on the hue range
% Hue values for red are typically near 0 or 1
red_mask = (hue_channel < 0.1 | hue_channel > 0.9) & ...
           (saturation_channel > 0.5) & ...
           (value_channel > 0.5);
cooper_mask = red_mask;
cooper_image = cooper_image(:,:,1);

cooper_image = logical(cooper_image);


% figure
delete('intermediate_cooper_figure.png');

hold off
end