function deformed_img = ApplyDeformation(img,d_ax,d_lat,is_mm,dpi)
% This function is meant to take in an input image, and apply a translation
% of each pixel in boh the x and y direction. This function was created
% with the intention of creating a deformed B-mode image based on the
% original B-mode image, and the axial and lateral displacements calculated
% from the FEA. Written in part of MESc completion. Created by Jonah Boutin
% on 12/05/2024. Note: This currently makes scattered data not an image.

% input_image = double(img); %Convert to type double
% input_image = uint8(img); %Convert to type uint8

% Need to convert from mm to pixels, if bool is true
if is_mm
    % Convert matrix to mm
    d_ax_mm = d_ax*100;
    d_lat_mm = d_lat*100;

    % Convert to pixels
    d_ax = (dpi/25.4).*d_ax_mm;
    d_lat = (dpi/25.4).*d_lat_mm;
end

input_image = img;
[rows, cols] = size(input_image); % Get image size
deformed_img = zeros(rows, cols); % Initialize the output image

for i = 1:rows
    for j = 1:cols
        % Move coordinates (backward mapping)
        i_prime = i - d_ax(i, j);
        j_prime = j - d_lat(i, j);

        % Ensure coordinates are within bounds
        if i_prime < 1 || i_prime > rows || j_prime < 1 || j_prime > cols
            continue; % Skip out-of-bounds pixels
        end

        % Get integer coordinates of the surrounding pixels
        i1 = floor(i_prime);
        i2 = ceil(i_prime);
        j1 = floor(j_prime);
        j2 = ceil(j_prime);

        % Handle edge cases by clamping indices
        i1 = max(1, min(rows, i1));
        i2 = max(1, min(rows, i2));
        j1 = max(1, min(cols, j1));
        j2 = max(1, min(cols, j2));

        % Compute the fractions
        s = i_prime - i1;
        t = j_prime - j1;

        % Get intensities of the four neighboring pixels
        I11 = input_image(i1, j1);
        I21 = input_image(i2, j1);
        I12 = input_image(i1, j2);
        I22 = input_image(i2, j2);

        % Perform bilinear interpolation
        new_gl = (1 - s) * (1 - t) * I11 + ...
                 s * (1 - t) * I21 + ...
                 (1 - s) * t * I12 + ...
                 s * t * I22;

        % Assign the computed intensity to the output image
        deformed_img(i, j) = new_gl;

    end
end

%deformed_img = uint8(deformed_img); % Convert back to uint8 for display