function I_blur = GenerateElastographyImage(Frame1,Frame2,visualize,Bmode,i)
% Some transducer params, these don't really matter
% FIELD II PARAMS
    params.probe.a_t = 1;
    params.probe.fc = 5;
    % params.probe.fc = 6.67;
    params.probe.fs = 50;
    % params.probe.fs = 40;
    params.L = 50;
    params.D = 60;

% KWAVE PARAMS
%     params.probe.a_t = 1;   % one-element TX (synthetic aperture); change if you use sub/full aperture
% params.probe.fc  = 5;   % MHz (matches f0 = 5e6)
% params.probe.fs  = 60;  % MHz (matches dt from k-Wave grid/time)
% params.L = 50;          % number of scanlines you want to beamform
% params.D = 62;          % number of elements in your array

    
    reconstruction_options = ReconOpts(0.01,false,false,'combined',10,3,false,'am2d_s');
  
    
    
    AM2D_disps = RunAM2D(Frame1, Frame2, params);


    
    [Disp_ax,Disp_lat,strainA,strainL,~]...
                  = prepdispsSTREAL(AM2D_disps.Axial(41:end-60,11:end-10),...
                  AM2D_disps.Lateral(41:end-60,11:end-10));

    assignin('base', 'Disp_ax', AM2D_disps.Axial)
    assignin('base', 'Disp_lat', AM2D_disps.Lateral)
    
    assignin('base', 'strainA', strainA)
    assignin('base', 'strainL', strainL)
    % if visualize

        % Create invisible figure
        % fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 1200]);
        figure
        subplot(2,3,1)
        imshow(Disp_ax, [])
        title("Axial Displacement (mm)")
        cb = colorbar;
        cb.FontSize = 12;

        subplot(2,3,2)
        imshow(strainA, [])
        title("Axial Strain")
        cb = colorbar;
        cb.FontSize = 12;

        subplot(2,3,3)
        imshow(Disp_lat, [])
        title("Lateral Displacement (mm)")
        cb = colorbar;
        cb.FontSize = 21;

        subplot(2,3,4)
        imshow(strainL(:,1:end-30), [])
        title("Lateral Strain")
        cb = colorbar;
        cb.FontSize = 12;

        % % Save the figure
        % % saveas(fig, sprintf('BreastClinData/%d_STREAL.png', row));  % PNG format
        % 
        % % Close the figure
        % % close(fig);
        %             % Set the super title for the figure (optional)
        % sgtitle(sprintf('STREAL_%d', i), 'FontSize', 28);
        % 
        % % Save the figure as an image
        % filename = sprintf('ClinicalResultsTest/STREAL_%d.png', i);
        % saveas(gcf, filename);
        % 
        % % Optionally, close the figure to avoid too many windows
        % close(gcf);


    % end
    
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

    disp('Reconstructing...')%,num2str(row)])
    
    % Perform Reconstructions
    reconstruction_result = RunReconstruction(reconstruction_options,boundary_conditions,YM_Image,strainA,strainL,Disp_ax,Disp_lat);
    % figure,imshow(reconstruction_result);
    %% Segmentation
% tx_i
    % I_gray = mat2gray(reconstruction_result);  % Normalize intensity to [0,1]
    % 
    % % Apply Gaussian blur to reduce noise
    % I_blur = imgaussfilt(I_gray, 2);
    % 
    % % Adaptive thresholding
    % bw = imbinarize(I_blur, 'adaptive', 'Sensitivity', 0.6); % Adjust sensitivity
    % 
    % % Remove small objects (noise) from binary image
    % bw_clean = bwareaopen(bw, 5000); % Remove small areas (< 500 pixels)
    % 
    % % Fill holes inside detected tumor regions
    % bw_filled = imfill(bw_clean, 'holes');
    % bw_edge = edge(bw_filled, 'Canny');
    % 
    % scale = 5*median(reconstruction_result, 'all');
    % reconstruction_result_overlaid = reconstruction_result;
    % reconstruction_result_overlaid(reconstruction_result_overlaid > scale) = scale;
    % 
    % reconstruction_result_overlaid = min(reconstruction_result_overlaid,scale);
    % reconstruction_result_overlaid_norm = (reconstruction_result_overlaid - min(reconstruction_result_overlaid(:))) / (max(reconstruction_result_overlaid(:)) - min(reconstruction_result_overlaid(:)));
    % if visualize
    %     figure,imshow(reconstruction_result_overlaid_norm),title('Elastography Image','FontSize',20)
    %     % reconstruction = repmat(reconstruction_result_overlaid_norm,[1,1,3]);
    %     % reconstruction(:,:,1) = reconstruction(:,:,1) .* ~bw_edge;
    %     % reconstruction(:,:,2) = reconstruction(:,:,2) .* ~bw_edge;
    %     % reconstruction(:,:,3) = reconstruction(:,:,1) + bw_edge*255;
    %     % 
    %     % figure,imshow(reconstruction),title('Elastography with overlaid tumor')
    % end

    % Normalize reconstruction to [0,1] based on robust percentile stretch
low = prctile(reconstruction_result(:), 2); 
high = prctile(reconstruction_result(:), 98);
reconstruction_result_clipped = min(max(reconstruction_result, low), high);
I_gray = mat2gray(reconstruction_result_clipped);  % Normalize

I_sharp = imsharpen(I_gray, 'Radius', 1, 'Amount', 1.5);
I_clahe = adapthisteq(I_sharp, 'NumTiles', [8 8], 'ClipLimit', 0.01);


% Slight Gaussian blur to reduce noise but preserve edges
I_blur = imgaussfilt(I_clahe, 0.7);

% Adaptive thresholding to segment likely tumor region
bw = imbinarize(I_blur, 'adaptive', 'Sensitivity', 0.55);  % Slightly more aggressive

% Remove small specks, fill holes
bw_clean = bwareaopen(bw, 200);       % keep smaller regions
bw_filled = imfill(bw_clean, 'holes');
bw_edge = edge(bw_filled, 'Canny');

% Overlay edge in red on grayscale image
reconstruction_rgb = repmat(I_clahe, 1, 1, 3);
reconstruction_rgb(:,:,1) = reconstruction_rgb(:,:,1) + 0.6 * bw_edge; % Red
reconstruction_rgb(:,:,2) = reconstruction_rgb(:,:,2) .* ~bw_edge;
reconstruction_rgb(:,:,3) = reconstruction_rgb(:,:,3) .* ~bw_edge;

% if visualize
%     figure;
%     imshow(I_blur(:,50:end),[]);
%     title('Elastography with Tumor Overlay', 'FontSize', 20);
% end


    % figure
    % subplot(2,4,1)
    % imshow(Disp_ax, [])
    % title("Axial Displacement (mm)", 'FontSize')
    % cb = colorbar;
    % % cb.FontSize = 20;
    % 
    % subplot(2,4,2)
    % imshow(strainA, [])
    % title("Axial Strain", 'FontSize')
    % cb = colorbar;
    % % cb.FontSize = 20;
    % 
    % subplot(2,4,3)
    % imshow(Disp_lat, [])
    % title("Lateral Displacement (mm)")
    % cb = colorbar;
    % % cb.FontSize = 20;
    % 
    % subplot(2,4,4)
    % imshow(strainL, [])
    % title("Lateral Strain")
    % cb = colorbar;
    % Create a 2x8 layout (image + colorbar pairs in top row, 4 images in bottom row)
% t = tiledlayout(2,8,'TileSpacing','compact','Padding','compact');
% 
% % --- Axial Displacement
% ax1 = nexttile(1);
% imshow(Disp_ax,[])
% title("Axial Displacement (mm)")
% cb1 = colorbar(ax1);
% cb1.Location = 'eastoutside';
% cb1.Layout.Tile = 2;  % assign to next tile
% 
% % --- Axial Strain
% ax2 = nexttile(3);
% imshow(strainA,[])
% title("Axial Strain")
% cb2 = colorbar(ax2);
% cb2.Location = 'eastoutside';
% cb2.Layout.Tile = 4;
% 
% % --- Lateral Displacement
% ax3 = nexttile(5);
% imshow(Disp_lat,[])
% title("Lateral Displacement (mm)")
% cb3 = colorbar(ax3);
% cb3.Location = 'eastoutside';
% cb3.Layout.Tile = 6;
% 
% % --- Lateral Strain
% ax4 = nexttile(7);
% imshow(strainL,[])
% title("Lateral Strain")
% cb4 = colorbar(ax4);
% cb4.Location = 'eastoutside';
% cb4.Layout.Tile = 8;
    % cb.FontSize = 20;
    % 
    % Save the figure
    % saveas(fig, sprintf('BreastClinData/%d_STREAL.png', row));  % PNG format
    
    % Close the figure
    % close(fig);
                % Set the super title for the figure (optional)
    % sgtitle(sprintf('STREAL_%d', i), 'FontSize', 28);
    
    % Save the figure as an image
    % filename = sprintf('ClinicalResultsTest/STREAL_%d.png', i);
    % saveas(gcf, filename);
    % 
    % % Optionally, close the figure to avoid too many windows
    % close(gcf);
    % subplot(2,4,5);
    % imshow(Bmode);
    % title('Bmode')
    % subplot(2,4,6);
    % imshow(reconstruction_result_overlaid_norm,[]);
    % title('Elastography');
    % subplot(2,4,7);
    % imshow(boundary_data);
    % title('Boundary');
    % img = zeros(220,220);
    % subplot(2,4,8);
    % imshow(img);
    % nexttile(9);
% imshow(Bmode);
% title('Bmode')
% 
% nexttile(11);
% imshow(reconstruction_result_overlaid_norm,[]);
% title('Elastography')
% 
% nexttile(13);
% imshow(boundary_data);
% title('Boundary')
% 
% nexttile(15);
% imshow(zeros(220,220));
% title('')
%     % gtitle(sprintf('Results_%d', i), 'FontSize', 28);
% % Big, clean canvas
% f = figure('Color','w','Units','pixels','Position',[100 100 1400 700]);
% 
% % 2 rows × 16 columns: each metric uses 3 cols for the image + 1 col for its colorbar
% t = tiledlayout(f, 2, 16, 'TileSpacing','compact', 'Padding','compact');
% 
% %% ---- Top row (images span 3 cols; colorbar sits in the next 1 col) ----
% % Axial Displacement
% ax1 = nexttile(1, [1 3]);  imshow(Disp_ax, []);  axis image off
% title('Axial Displacement (mm)')
% cb1 = colorbar(ax1); cb1.Location = 'eastoutside'; cb1.Layout.Tile = 4;
% 
% % Axial Strain
% ax2 = nexttile(5, [1 3]);  imshow(strainA, []);  axis image off
% title('Axial Strain')
% cb2 = colorbar(ax2); cb2.Location = 'eastoutside'; cb2.Layout.Tile = 8;
% 
% % Lateral Displacement
% ax3 = nexttile(9, [1 3]);  imshow(Disp_lat, []);  axis image off
% title('Lateral Displacement (mm)')
% cb3 = colorbar(ax3); cb3.Location = 'eastoutside'; cb3.Layout.Tile = 12;
% 
% % Lateral Strain
% ax4 = nexttile(13, [1 3]); imshow(strainL, []);  axis image off
% title('Lateral Strain')
% cb4 = colorbar(ax4); cb4.Location = 'eastoutside'; cb4.Layout.Tile = 16;
% 
% %% ---- Bottom row (four panels, each spans 4 columns) ----
% ax = nexttile(17, [1 4]); imshow(Bmode, []);                       axis image off; title('Bmode');
% ax = nexttile(21, [1 4]); imshow(reconstruction_result_overlaid_norm, []); axis image off; title('Elastography');
% ax = nexttile(25, [1 4]); imshow(boundary_data, []);               axis image off; title('Boundary');
% ax = nexttile(29, [1 4]); imshow(zeros(220,220));                  axis image off;

% figure;
subplot(2,3,5);
imshow(Bmode);
title('Bmode')
% 
subplot(2,3,6);
imshow(I_blur,[]);
title('Elastography')
% 
% subplot(1,3,3);
% imshow(boundary_data);
% title('Boundary')
% 
    % filename = sprintf('ClinData_7/Result_%d.png', i);
    % saveas(gcf, filename);
    % exportgraphics(gcf, filename, 'Resolution', 300)
    
    % Optionally, close the figure to avoid too many windows
    % close(gcf);