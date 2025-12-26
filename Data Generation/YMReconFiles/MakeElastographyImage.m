% This script is to genereate YM Reconstruction images to be used to train
% the ML network to output elastography images.
% Created by Jonah Boutin on 03/26/2025
clc,clear
% FrameData = load("TestFrames.mat").TestFrames;
testing_folder = "C:\Users\1bout\OneDrive\SamaniLab\MESc Files\TestRFData\";
testing_files = dir(fullfile(testing_folder,"*.mat"));
for i = 2%:length(testing_files)

    file_path = fullfile(testing_folder,testing_files(i).name);
    FrameData = load(file_path).result;

    Frame1 = FrameData.Frame1;
    Frame2 = FrameData.Frame2;
    % tumor_mask = FrameData.image_information.tumor_mask;
    % cooper_mask = FrameData.image_information.cooper_mask;
    tumor_mask = FrameData.Frame1;
    cooper_mask = FrameData.Frame2;
    
    % Some transducer params, these don't really matter
    params.probe.a_t = 1;
    params.probe.fc = 5;
    params.probe.fs = 50;
    params.L = 50;
    params.D = 60;
    
    reconstruction_options = ReconOpts(0.01,false,true,'combined',10,5,true,'am2d_s');
    
    
    AM2D_disps = RunAM2D(Frame1, Frame2, params);
    
    [Disp_ax,Disp_lat,strainA,strainL,~]...
                  = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
                  AM2D_disps.Lateral(41:end-60,11:end-10));
    
    % First assign the boundary conditions by extracting them from the
    % displacements calculated
    % Boundary Conditions
    boundary_conditions = clib.FEM_Interface.BoundaryStruct;
    
    boundary_conditions.top_axial = ConvertPXToMM(Disp_ax(1,:));   
    boundary_conditions.bottom_axial = ConvertPXToMM(Disp_ax(end,:));
    
    boundary_conditions.top_lateral = ConvertPXToMM(Disp_lat(1,:));   
    boundary_conditions.bottom_lateral = ConvertPXToMM(Disp_lat(end,:));
    
    boundary_conditions.right_axial = ConvertPXToMM(Disp_ax(:,end)');
    boundary_conditions.left_axial = ConvertPXToMM(Disp_ax(:,1)');
    
    boundary_conditions.right_lateral = ConvertPXToMM(Disp_lat(:,end)');
    boundary_conditions.left_lateral = ConvertPXToMM(Disp_lat(:,1)');
    
    %% Young's Modulus calculations and image recreation
    
    % Initialize first guess YM field
    % YM_Image = strainA*1000;
    YM_Image = 3000*ones(size(strainA));
    
    % Perform Reconstructions
    reconstruction_result = RunReconstruction(reconstruction_options,boundary_conditions,YM_Image,strainA,strainL,Disp_ax,Disp_lat);
    
    %% Segmentation
    
    I_gray = mat2gray(reconstruction_result);  % Normalize intensity to [0,1]
    
    % Apply Gaussian blur to reduce noise
    I_blur = imgaussfilt(I_gray, 2);
    
    % Adaptive thresholding
    bw = imbinarize(I_blur, 'adaptive', 'Sensitivity', 0.6); % Adjust sensitivity
    
    % Remove small objects (noise) from binary image
    bw_clean = bwareaopen(bw, 5000); % Remove small areas (< 500 pixels)
    
    % Fill holes inside detected tumor regions
    bw_filled = imfill(bw_clean, 'holes');
    bw_edge = edge(bw_filled, 'Canny');
    
    scale = 5*median(reconstruction_result, 'all');
    reconstruction_result_overlaid = reconstruction_result;
    reconstruction_result_overlaid(reconstruction_result_overlaid > scale) = scale;
    
    reconstruction_result_overlaid = min(reconstruction_result_overlaid,scale);
    reconstruction_result_overlaid_norm = (reconstruction_result_overlaid - min(reconstruction_result_overlaid(:))) / (max(reconstruction_result_overlaid(:)) - min(reconstruction_result_overlaid(:)));
    
    reconstruction = repmat(reconstruction_result_overlaid_norm,[1,1,3]);
    reconstruction(:,:,1) = reconstruction(:,:,1) .* ~bw_edge;
    reconstruction(:,:,2) = reconstruction(:,:,2) .* ~bw_edge;
    reconstruction(:,:,3) = reconstruction(:,:,1) + bw_edge*255;
    
    figure,imshow(reconstruction),title('Elastography with overlaid tumor')
    
    % Create a structural element for closing (a disk-shaped element)
    se = strel('disk', 45);  % You can adjust the size (5 here) as needed
    
    % Perform the closing operation
    bw_filled_tumor = imclose(bw_edge, se);
    
    % Display the result
    % figure, imshow(bw_filled_tumor);
end

