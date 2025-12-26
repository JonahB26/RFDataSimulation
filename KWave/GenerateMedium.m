% function [sound_speed_map_pre, density_map_pre, Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, num_scan_lines, elem_width_gdpts, bg_mean, bg_std, c0, rho0)
% % Nx_tot = Nx;
% % Ny_tot = Ny + num_scan_lines*elem_width_gdpts;   % wide enough to slide
% % noise = randn([Nx_tot, Ny_tot]);
% % 
% % % Physical extents of the full grid
% % Xmin = 0;                Xmax = Nx_tot*dx;
% % Ymin = 0;                Ymax = Ny_tot*dy;
% % 
% % background_map   = bg_mean + bg_std*noise;
% % sound_speed_map  = c0   * background_map;
% % density_map      = rho0 * background_map;
% % 
% % output = load(filename).output;
% % % % TumorArea = load(filename).TumorArea;
% % % % inc256 = imresize(TumorArea,[256 256]);
% % % inc256 = output.images.tumor_mask;
% % % 
% % % % % (A) Use native pixel size (no scaling), or
% % % % min_width = 20e-3;
% % % % max_width = 35e-3;
% % % % target_width_m = min_width + (max_width - min_width)*rand();
% % % % % target_width_m  = 20e-3;    % desired lateral width of the inclusion (meters)
% % % % target_height_m = [];       % [] keeps aspect ratio; set if you want explicit
% % % % 
% % % %     % Compute desired size in grid samples from physical size
% % % %     if isempty(target_height_m), target_height_m = target_width_m * (size(inc256,1)/size(inc256,2)); end
% % % %     tgt_w_px = max(1, round(target_width_m  / dy));   % columns (lateral)
% % % %     tgt_h_px = max(1, round(target_height_m / dx));   % rows    (axial)
% % % %     inc_resized = imresize(inc256, [tgt_h_px, tgt_w_px], 'nearest');
% % % % 
% % % % 
% % % % inc_resized = logical(inc_resized);   % ensure logical mask
% % % % [hI, wI] = size(inc_resized);
% % % % 
% % % % % Choose a placement region (a box) where the inclusion must fit entirely
% % % % % Define a physical placement box (you can reuse the same as above or make larger)
% % % % place_box_w_m = 15e-3;  place_box_h_m = 15e-3;
% % % % place_cx = (Xmax - Xmin)/2;           % center (meters)
% % % % place_cy = (Ymax - Ymin)/2;
% % % % 
% % % % % Convert feasible top-left indices in samples so the block stays inside
% % % % % Allowed i (rows / axial) and j (cols / lateral) index ranges:
% % % % i_lo = 1;
% % % % i_hi = Nx_tot - hI + 1;
% % % % j_lo = 1;
% % % % j_hi = Ny_tot - wI + 1;
% % % % 
% % % % % If you want to limit to a smaller physical placement box:
% % % % i_box_lo = max(i_lo, floor((place_cx - place_box_h_m/2)/dx)); % NOTE: rows map to x-depth spacing dx
% % % % i_box_hi = min(i_hi, ceil( (place_cx + place_box_h_m/2)/dx));
% % % % j_box_lo = max(j_lo, floor((place_cy - place_box_w_m/2)/dy)); % columns map to y-lateral spacing dy
% % % % j_box_hi = min(j_hi, ceil( (place_cy + place_box_w_m/2)/dy));
% % % % 
% % % % % Fallback if box too tight:
% % % % if i_box_lo > i_box_hi, i_box_lo = i_lo; i_box_hi = i_hi; end
% % % % if j_box_lo > j_box_hi, j_box_lo = j_lo; j_box_hi = j_hi; end
% % % % 
% % % % % Sample a random top-left index within the allowed region
% % % % i0 = randi([i_box_lo, i_box_hi]);
% % % % j0 = randi([j_box_lo, j_box_hi]);
% % % % 
% % % % % Build a full-size logical mask and paste the resized inclusion into it
% % % % mask_img = false(Nx_tot, Ny_tot);
% % % % mask_img(i0:(i0+hI-1), j0:(j0+wI-1)) = inc_resized;
% % % % ------------ 1) Resize to target width, but clamp to grid ------------
% % % inc256 = logical(inc256);
% % % [h0, w0] = size(inc256);
% % % 
% % % min_width = 20e-3; max_width = 25e-3;
% % % target_width_m = min_width + (max_width - min_width)*rand();
% % % 
% % % tgt_w_px = max(1, round(target_width_m / dy));               % columns (lateral)
% % % tgt_h_px = max(1, round((target_width_m * (h0/w0)) / dx));   % rows (axial)
% % % 
% % % % clamp to grid so it always fits
% % % tgt_w_px = min(tgt_w_px, Ny_tot);
% % % tgt_h_px = min(tgt_h_px, Nx_tot);
% % % 
% % % inc_resized = imresize(inc256, [tgt_h_px, tgt_w_px], 'nearest');
% % % [hI, wI] = size(inc_resized);
% % % 
% % % % ------------ 2) Global valid top-left ranges (must be integers, lo<=hi) ------------
% % % i_lo = 1;                   i_hi = max(i_lo, Nx_tot - hI + 1);
% % % j_lo = 1;                   j_hi = max(j_lo, Ny_tot - wI + 1);
% % % 
% % % % ------------ 3) Optional placement box — only if it can contain the mask ------------
% % % place_box_w_m = 15e-3;  place_box_h_m = 15e-3;  % NOTE: must be >= mask size
% % % place_cx = (Xmax - Xmin)/2;   % axial center (meters)
% % % place_cy = (Ymax - Ymin)/2;   % lateral center (meters)
% % % 
% % % % convert box to index ranges
% % % i_box_lo = max(i_lo, floor((place_cx - place_box_h_m/2)/dx));
% % % i_box_hi = min(i_hi, ceil( (place_cx + place_box_h_m/2)/dx));
% % % j_box_lo = max(j_lo, floor((place_cy - place_box_w_m/2)/dy));
% % % j_box_hi = min(j_hi, ceil( (place_cy + place_box_w_m/2)/dy));
% % % 
% % % % If the box is smaller than the mask or collapses, ignore the box
% % % box_can_fit = (place_box_w_m >= wI*dy) && (place_box_h_m >= hI*dx) && ...
% % %               (i_box_lo <= i_box_hi) && (j_box_lo <= j_box_hi);
% % % 
% % % if box_can_fit
% % %     Imin = i_box_lo;  Imax = i_box_hi;
% % %     Jmin = j_box_lo;  Jmax = j_box_hi;
% % % else
% % %     Imin = i_lo;      Imax = i_hi;
% % %     Jmin = j_lo;      Jmax = j_hi;
% % % end
% % % 
% % % % ------------ 4) Safe sampling (no invalid randi ranges) ------------
% % % if Imin > Imax,  i0 = round((i_lo + i_hi)/2); else, i0 = randi([Imin, Imax]); end
% % % if Jmin > Jmax,  j0 = round((j_lo + j_hi)/2); else, j0 = randi([Jmin, Jmax]); end
% % % 
% % % % ------------ 5) Paste ------------
% % % mask_img = false(Nx_tot, Ny_tot);
% % % mask_img(i0:(i0+hI-1), j0:(j0+wI-1)) = inc_resized;
% % % 
% % % mask_img = output.images.tumor_mask;
% % mask_img = logical(output.images.tumor_mask);
% % 
% % if strcmp(output.tumor_info.label,'malignant')
% %     inc_mean = 1570; inc_std = 75;
% % else 
% %     inc_mean = 1600; inc_std = 75;
% % end
% % 
% % % Apply contrast in exactly the same way as for discs
% % % jitter = randn([Nx_tot, Ny_tot]);
% % c_s = inc_mean + inc_std*noise;
% % c_s = min(max(c_s, 1500), 1620);
% % rho = c_s/1.5;
% % 
% % sound_speed_map(mask_img) = c_s(mask_img);
% % density_map(mask_img)     = rho(mask_img);
% % 
% % % Add coopers
% % % cooper_mask = imresize(output.images.cooper_mask, [Nx_tot, Ny_tot]);
% % cooper_mask = output.images.cooper_mask;
% % c_sc = 1610 + 25*noise;
% % c_sc = min(max(c_sc, 1550), 1650);
% % rho_sc = c_sc/1.5;
% % 
% % sound_speed_map(cooper_mask) = c_sc(cooper_mask);
% % density_map(cooper_mask) = rho_sc(cooper_mask);
% % 
% % % Apply displacements
% % axial_disp = output.disps.axial_disp / 1000;
% % lateral_disp = output.disps.lateral_disp / 1000;
% % 
% % % axial_disp = imresize(axial_disp, [Nx_tot, Ny_tot]);
% % % lateral_disp = imresize(lateral_disp, [Nx_tot, Ny_tot]);
% 
% % -------------------------------------------------------------------------
% % Generate baseline (pre-compression) and deformed (post-compression)
% % medium property maps for elastography simulations.
% % -------------------------------------------------------------------------
% 
% %% --- 1) Initialize grid and background noise ---
% Nx_tot = Nx;
% Ny_tot = Ny + num_scan_lines * elem_width_gdpts;    % wide enough to slide laterally
% noise = randn([Nx_tot, Ny_tot]);
% 
% Xmin = 0; Xmax = Nx_tot * dx;
% Ymin = 0; Ymax = Ny_tot * dy;
% 
% background_map = bg_mean + bg_std * noise;
% sound_speed_map_pre = c0   * background_map;
% density_map_pre     = rho0 * background_map;
% 
% %% --- 2) Load masks and displacements from FEA output ---
% output = load(filename).output;
% 
% mask_img    = logical(output.images.tumor_mask);
% cooper_mask = logical(output.images.cooper_mask);
% axial_disp   = output.disps.axial_disp   / 1000;  % [m]
% lateral_disp = output.disps.lateral_disp / 1000;  % [m]
% 
% % resize displacements and masks if needed
% if ~isequal(size(axial_disp), [Nx_tot, Ny_tot])
%     axial_disp   = imresize(axial_disp,   [Nx_tot, Ny_tot], 'bicubic');
%     lateral_disp = imresize(lateral_disp, [Nx_tot, Ny_tot], 'bicubic');
% end
% if ~isequal(size(mask_img), [Nx_tot, Ny_tot])
%     mask_img = imresize(mask_img, [Nx_tot, Ny_tot], 'nearest');
% end
% if ~isequal(size(cooper_mask), [Nx_tot, Ny_tot])
%     cooper_mask = imresize(cooper_mask, [Nx_tot, Ny_tot], 'nearest');
% end
% 
% %% --- 3) Assign tissue contrasts ---
% if strcmp(output.tumor_info.label,'malignant')
%     inc_mean = 1570; inc_std = 75;
% else
%     inc_mean = 1600; inc_std = 75;
% end
% 
% % Inclusion contrast
% c_inc  = inc_mean + inc_std * noise;
% c_inc  = min(max(c_inc, 1500), 1620);
% rho_inc = c_inc / 1.5;
% 
% sound_speed_map_pre(mask_img) = c_inc(mask_img);
% density_map_pre(mask_img)     = rho_inc(mask_img);
% 
% % Cooper's ligaments
% c_coop  = 1610 + 25 * noise;
% c_coop  = min(max(c_coop, 1550), 1650);
% rho_coop = c_coop / 1.5;
% 
% sound_speed_map_pre(cooper_mask) = c_coop(cooper_mask);
% density_map_pre(cooper_mask)     = rho_coop(cooper_mask);
% 
% %% --- 4) Apply deformation to create POST-compression maps ---
% % Build grid coordinates (in meters)
% [xg, yg] = ndgrid((0:Nx_tot-1)*dx, (0:Ny_tot-1)*dy);
% 
% % Inverse mapping (where each new pixel came from)
% x_src = xg - axial_disp;
% y_src = yg - lateral_disp;
% 
% % Clip to valid region
% x_src = min(max(x_src, 0), (Nx_tot-1)*dx);
% y_src = min(max(y_src, 0), (Ny_tot-1)*dy);
% 
% % Interpolants for inverse warping
% Fc_pre   = griddedInterpolant(xg, yg, sound_speed_map_pre, 'linear', 'nearest');
% Frho_pre = griddedInterpolant(xg, yg, density_map_pre,     'linear', 'nearest');
% 
% sound_speed_map_post = Fc_pre(x_src, y_src);
% density_map_post     = Frho_pre(x_src, y_src);
% 
% 
% 
% 
% 
% figure;
% subplot(121);
% imshow(axial_disp,[]);title('Axial Disp')
% subplot(122);
% imshow(lateral_disp,[]);title('Lateral Disp')
% 
% 
% 
% figure;
% subplot(121);
% imagesc((0:num_scan_lines*elem_width_gdpts-1)*dy*1e3, (0:Nx_tot-1)*dx*1e3, ...
%         sound_speed_map_pre(:, 1 + Ny/2 : end - Ny/2));
% axis image; colormap(gray); set(gca,'YLim',[5,40]);
% title('Scattering Phantom - PRE'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
% 
% subplot(122)
% imagesc((0:num_scan_lines*elem_width_gdpts-1)*dy*1e3, (0:Nx_tot-1)*dx*1e3, ...
%         sound_speed_map_post(:, 1 + Ny/2 : end - Ny/2));
% axis image; colormap(gray); set(gca,'YLim',[5,40]);
% title('Scattering Phantom - POST'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
% end

function [sound_speed_map_pre, density_map_pre, ...
          sound_speed_map_post, density_map_post, ...
          Nx_tot, Ny_tot] = GenerateMedium(filename, Nx, Ny, dx, dy, ...
                                           num_scan_lines, elem_width_gdpts, ...
                                           bg_mean, bg_std, c0, rho0)
% -------------------------------------------------------------------------
% Generate baseline (pre-compression) and deformed (post-compression)
% medium property maps for elastography simulations.
% -------------------------------------------------------------------------

%% --- 1) Initialize grid and background noise ---
Nx_tot = Nx;
Ny_tot = Ny + num_scan_lines * elem_width_gdpts;    % wide enough to slide laterally
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ny_tot = Ny;    % wide enough to slide laterally
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

noise = randn([Nx_tot, Ny_tot]);

Xmin = 0; Xmax = Nx_tot * dx;
Ymin = 0; Ymax = Ny_tot * dy;

background_map = bg_mean + bg_std * noise;
sound_speed_map_pre = c0   * background_map;
density_map_pre     = rho0 * background_map;



%% --- 2) Load masks and displacements from FEA output ---
output = load(filename).output;

mask_img    = logical(output.images.tumor_mask);
cooper_mask = logical(output.images.cooper_mask);
axial_disp   = output.disps.axial_disp   / 1000;  % [m]
lateral_disp = output.disps.lateral_disp / 1000;  % [m]

% resize displacements and masks if needed
if ~isequal(size(axial_disp), [Nx_tot, Ny_tot])
    axial_disp   = imresize(axial_disp,   [Nx_tot, Ny_tot], 'bicubic');
    lateral_disp = imresize(lateral_disp, [Nx_tot, Ny_tot], 'bicubic');
end
if ~isequal(size(mask_img), [Nx_tot, Ny_tot])
    mask_img = imresize(mask_img, [Nx_tot, Ny_tot], 'nearest');
end
if ~isequal(size(cooper_mask), [Nx_tot, Ny_tot])
    cooper_mask = imresize(cooper_mask, [Nx_tot, Ny_tot], 'nearest');
end

%% --- 3) Assign tissue contrasts ---
if strcmp(output.tumor_info.label,'malignant')
    inc_mean = 1570; inc_std = 75;
else
    inc_mean = 1600; inc_std = 75;
end

% Inclusion contrast
c_inc  = inc_mean + inc_std * noise;
c_inc  = min(max(c_inc, 1500), 1620);
rho_inc = c_inc / 1.5;

sound_speed_map_pre(mask_img) = c_inc(mask_img);
density_map_pre(mask_img)     = rho_inc(mask_img);

% Cooper's ligaments
c_coop  = 1610 + 25 * noise;
c_coop  = min(max(c_coop, 1550), 1650);
rho_coop = c_coop / 1.5;

sound_speed_map_pre(cooper_mask) = c_coop(cooper_mask);
density_map_pre(cooper_mask)     = rho_coop(cooper_mask);

phantom_data = struct();
phantom_data.sound_speed_map = sound_speed_map_pre;
phantom_data.density_map = density_map_pre;
save('PhantomData.mat',"phantom_data");

%% --- 4) Apply deformation to create POST-compression maps ---
% Build grid coordinates (in meters)
[xg, yg] = ndgrid((0:Nx_tot-1)*dx, (0:Ny_tot-1)*dy);

% Inverse mapping (where each new pixel came from)
x_src = xg - axial_disp;
y_src = yg - lateral_disp;

% Clip to valid region
x_src = min(max(x_src, 0), (Nx_tot-1)*dx);
y_src = min(max(y_src, 0), (Ny_tot-1)*dy);

% Interpolants for inverse warping
Fc_pre   = griddedInterpolant(xg, yg, sound_speed_map_pre, 'linear', 'nearest');
Frho_pre = griddedInterpolant(xg, yg, density_map_pre,     'linear', 'nearest');

sound_speed_map_post = Fc_pre(x_src, y_src);
density_map_post     = Frho_pre(x_src, y_src);

%% --- 5) Optional visualization sanity checks ---
figure('Name','Deformation field');
subplot(1,2,1);
imagesc(yg(1,:)*1e3, xg(:,1)*1e3, axial_disp*1e3); axis image ij;
title('Axial displacement [mm]'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
subplot(1,2,2);
imagesc(yg(1,:)*1e3, xg(:,1)*1e3, lateral_disp*1e3); axis image ij;
title('Lateral displacement [mm]'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
colormap jet;

figure('Name','Sound Speed: Pre vs Post');
subplot(1,2,1);
imagesc(yg(1,:)*1e3, xg(:,1)*1e3, sound_speed_map_pre); axis image ij;
title('Sound Speed (Pre)'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
subplot(1,2,2);
imagesc(yg(1,:)*1e3, xg(:,1)*1e3, sound_speed_map_post); axis image ij;
title('Sound Speed (Post)'); xlabel('Lateral [mm]'); ylabel('Depth [mm]');
colormap gray;

end
