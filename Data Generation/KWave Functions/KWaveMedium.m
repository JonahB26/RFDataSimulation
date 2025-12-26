% classdef KWaveMedium
%   %KWaveMedium Build k-Wave medium maps with placed inclusion and Cooper mask.
%   %
%   % Usage:
%   %   med = KWaveMedium(arrayObj, Ms, cooper_mask, ...
%   %         'z_mask_top_mm', 6, 'y_target_mm', 0, ...
%   %         'c_cooper', 1650, 'rho_cooper', 1100, 'alpha_cooper', 0.35);
%   %
%   %   medium = med.getmediumStruct();
%   %   imagesc(med.density), axis image, colorbar
% 
%   properties (SetAccess = immutable)
%     arrayObj
%     inclusionMask logical        % raw inclusion mask (Ms) before placement
%     cooperMask    logical        % Nx×Ny logical (true where Cooper's exists)
%   end
% 
%   % placement
%   properties (SetAccess = immutable)
%     z_mask_top_mm (1,1) double = 6
%     y_target_mm   (1,1) double = 0
%   end
% 
%   % background material defaults
%   properties (SetAccess = immutable)
%     c_bg   (1,1) double = 1540
%     rho_bg (1,1) double = 1000
%     alpha_power (1,1) double = 1.5
%     alpha_bg    (1,1) double = 0.15    % dB/(MHz^alpha_power*cm)
%     lower_alpha_inside (1,1) logical = true
%     alpha_inside_scale (1,1) double = 2.0
%   end
% 
%   % speckle / texture (outside vs inclusion)
%   properties (SetAccess = immutable)
%     sig_out (1,1) double = 1.6
%     sig_in  (1,1) double = 0.6
%     rho_out (1,1) double = 0.03
%     rho_in  (1,1) double = 0.30
%     c_out   (1,1) double = 0.0015
%     c_in    (1,1) double = 0.015
%   end
% 
%   % Cooper's material fixed properties (you can override)
%   properties (SetAccess = immutable)
%     c_cooper   (1,1) double = 1650
%     rho_cooper (1,1) double = 1100
%     alpha_cooper (1,1) double = 0.3
%   end
% 
%   % outputs
%   properties (SetAccess = private)
%     inc_mask logical            % Nx×Ny placed inclusion mask
%     density  single             % Nx×Ny
%     sound_speed single          % Nx×Ny
%     alpha_coeff single          % Nx×Ny
%   end
% 
%   methods
%     function obj = KWaveMedium(arrayObj, Ms, cooper_mask, varargin)
%       obj.arrayObj     = arrayObj;
%       obj.inclusionMask = logical(Ms);
% 
%       % Cooper mask can be empty or logical Nx×Ny
%       if isempty(cooper_mask)
%         obj.cooperMask = false(arrayObj.Nx, arrayObj.Ny);
%       else
%         cm = logical(cooper_mask);
%         % if size already matches grid, accept; else error (be explicit)
%         if isequal(size(cm), [arrayObj.Nx, arrayObj.Ny])
%           obj.cooperMask = cm;
%         else
%           error('cooper_mask must be Nx-by-Ny to match the grid.');
%         end
%       end
% 
%       % Apply name/value overrides
%       for k = 1:2:numel(varargin)
%         name = varargin{k}; val = varargin{k+1};
%         if isprop(obj, name), obj.(name) = val;
%         else, error('Unknown option: %s', name);
%         end
%       end
% 
%       % Short-hands
%       Nx = arrayObj.Nx; Ny = arrayObj.Ny;
%       dx = arrayObj.dx; % dy unused below but available
%       x_src = arrayObj.x_src;
%       y_vec = arrayObj.kgrid.y_vec;
% 
%       % ---- 1) Place inclusion Ms into grid (obj.inc_mask) ----
%       Hm = size(Ms,1)*dx;
%       cx_from_top = (obj.z_mask_top_mm/1e3) + Hm/2;
%       ix_t = x_src + round(cx_from_top / dx);
%       iy_t = round(interp1(y_vec, 1:Ny, obj.y_target_mm/1e3, 'nearest', 'extrap'));
% 
%       hx = floor(size(Ms,1)/2);
%       hy = floor(size(Ms,2)/2);
%       ix_t = min(max(ix_t, 1 + hx), Nx - hx);
%       iy_t = min(max(iy_t, 1 + hy), Ny - hy);
% 
%       top  = ix_t - hx; left = iy_t - hy;
%       rIdx = max(1, top)  : min(Nx, top  + size(Ms,1) - 1);
%       cIdx = max(1, left) : min(Ny, left + size(Ms,2) - 1);
%       srcR = (rIdx - top + 1);
%       srcC = (cIdx - left + 1);
% 
%       obj.inc_mask = false(Nx,Ny);
%       obj.inc_mask(rIdx, cIdx) = Ms(srcR, srcC);
% 
%       % ---- 2) Background + Inclusion speckle (same as before) ----
%       g_out = fspecial('gaussian', [7 7], obj.sig_out);
%       g_in  = fspecial('gaussian', [7 7], obj.sig_in);
% 
%       n_r_out = conv2(randn(Nx,Ny), g_out, 'same');
%       n_r_in  = conv2(randn(Nx,Ny), g_in , 'same');
%       n_c_out = conv2(randn(Nx,Ny), g_out, 'same');
%       n_c_in  = conv2(randn(Nx,Ny), g_in , 'same');
% 
%       zn = @(A) ( (A - mean(A(:))) ./ std(A(:)) );
%       n_r_out = zn(n_r_out); n_r_in = zn(n_r_in);
%       n_c_out = zn(n_c_out); n_c_in = zn(n_c_in);
% 
%       n_r = n_r_out; n_r(obj.inc_mask) = n_r_in(obj.inc_mask);
%       n_c = n_c_out; n_c(obj.inc_mask) = n_c_in(obj.inc_mask);
% 
%       Ar = obj.rho_out*ones(Nx,Ny,'single'); Ar(obj.inc_mask) = obj.rho_in;
%       Ac = obj.c_out  *ones(Nx,Ny,'single'); Ac(obj.inc_mask) = obj.c_in;
% 
%       rho_map = single(obj.rho_bg) .* (1 + Ar .* single(n_r));
%       c_map   = single(obj.c_bg)   .* (1 + Ac .* single(n_c));
% 
%       % clip for stability
%       rho_map = min(max(rho_map, single(obj.rho_bg*(1-0.25))), single(obj.rho_bg*(1+0.25)));
%       c_map   = min(max(c_map,   single(obj.c_bg  *(1-0.02))), single(obj.c_bg  *(1+0.02)));
% 
%       % attenuation (bg + optionally lower inside)
%       alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
%       if obj.lower_alpha_inside
%         alpha(obj.inc_mask) = single(obj.alpha_inside_scale * obj.alpha_bg);
%       end
% 
%       % ---- 3) Cooper's overlay (third material, highest priority) ----
%       if any(obj.cooperMask(:))
%         % overwrite with fixed properties
%         idx = obj.cooperMask;
%         c_map(idx)     = single(obj.c_cooper);
%         rho_map(idx)   = single(obj.rho_cooper);
%         alpha(idx)     = single(obj.alpha_cooper);
%       end
%       % Priority order is: Cooper's > Inclusion > Background
% 
%       % store
%       obj.density     = rho_map;
%       obj.sound_speed = c_map;
%       obj.alpha_coeff = alpha;
%       obj.alpha_power = obj.alpha_power;
%     end
% 
%     function medium = getmediumStruct(obj)
%       medium = struct();
%       medium.sound_speed = obj.sound_speed;
%       medium.density     = obj.density;
%       medium.alpha_power = obj.alpha_power;
%       medium.alpha_coeff = obj.alpha_coeff;
%       % figure;imshow(medium.density,[]);title('Dens')
%       % figure;imshow(medium.sound_speed,[]);title('ss')
%       % figure;imshow(medium.alpha_power,[]);title('power')
%       % figure;imshow(medium.alpha_coeff,[]);title('alpha')
%     end
%   end
% end

% classdef KWaveMedium
%   %KWaveMedium Build baseline AND deformed k-Wave medium maps.
%   % - Reuses the SAME speckle realization for pre/post
%   % - Adds mean contrast in the inclusion (rho +2%, c +3%)
%   % - Keeps variance small and uniform to reduce decorrelation
%   % - Higher attenuation in stiff inclusion
%   % - Optionally adds tiny texture to Cooper's ligaments
%   %
%   % Usage:
%   %   med = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, ...
%   %         'z_mask_top_mm', 6, 'y_target_mm', 0, ...
%   %         'rngSeed', 42);
%   %
%   %   [med0, med1] = med.getMediumStructs();   % baseline, deformed
%   %   imagesc(med.density), axis image, colorbar
% 
%   %% -------------------- immutable inputs --------------------
%   properties (SetAccess = immutable)
%     arrayObj
%     inclusionMask logical         % raw Ms before placement
%     cooperMask    logical         % Nx-by-Ny logical (true where Cooper's exists)
%     simresult                         % struct with axial_disp, lateral_disp (mm or pre-converted)
%   end
% 
%   % placement
%   properties (SetAccess = immutable)
%     z_mask_top_mm (1,1) double = 6
%     y_target_mm   (1,1) double = 0
%   end
% 
%   % background material defaults (physical-ish)
%   properties (SetAccess = immutable)
%     c_bg   (1,1) double = 1540
%     rho_bg (1,1) double = 1000
%     alpha_power (1,1) double = 1.5
%     alpha_bg    (1,1) double = 0.6     % dB/(MHz^alpha_power*cm)
%   end
% 
%   % inclusion contrasts (kept modest to preserve correlation)
%   properties (SetAccess = immutable)
%     dc_mean  (1,1) double = +0.03      % +3% c inside
%     dr_mean  (1,1) double = +0.02      % +2% rho inside
%     sigma_c  (1,1) double = 0.005      % 0.5% c everywhere
%     sigma_r  (1,1) double = 0.02       % 2%   rho everywhere
%     alpha_inside_scale (1,1) double = 1.25  % 25% higher α in stiff inclusion
%   end
% 
%   % Cooper's fixed properties + tiny texture
%   properties (SetAccess = immutable)
%     c_cooper   (1,1) double = 1650
%     rho_cooper (1,1) double = 1100
%     alpha_cooper (1,1) double = 0.7    % usually high in fibrous tissue
%     cooper_sigma_c (1,1) double = 0.003
%     cooper_sigma_r (1,1) double = 0.01
%   end
% 
%   % RNG / reproducibility
%   properties (SetAccess = immutable)
%     rngSeed   % [] = no change; otherwise sets rng(seed) at construction
%   end
% 
%   %% -------------------- outputs (baseline + deformed) --------------------
%   properties (SetAccess = private)
%     % placed binary masks
%     inc_mask   logical
%     coop_mask  logical
% 
%     % baseline maps
%     density      single
%     sound_speed  single
%     alpha_coeff  single
% 
%     % deformed maps
%     density_def     single
%     sound_speed_def single
%     alpha_coeff_def single
%     inc_mask_def    logical
%     coop_mask_def   logical
%   end
% 
%   methods
%     function obj = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, varargin)
%       % --------- store inputs ----------
%       obj.arrayObj      = arrayObj;
%       obj.inclusionMask = logical(Ms);
%       obj.simresult     = simresult;
% 
%       if isempty(cooper_mask)
%         obj.cooperMask = false(arrayObj.Nx, arrayObj.Ny);
%       else
%         cm = logical(cooper_mask);
%         if isequal(size(cm), [arrayObj.Nx, arrayObj.Ny])
%           obj.cooperMask = cm;
%         else
%           error('cooper_mask must be Nx-by-Ny to match the grid.');
%         end
%       end
% 
%       % --------- parse overrides ----------
%       p = inputParser;
%       p.addParameter('z_mask_top_mm', obj.z_mask_top_mm);
%       p.addParameter('y_target_mm',   obj.y_target_mm);
%       p.addParameter('c_bg',          obj.c_bg);
%       p.addParameter('rho_bg',        obj.rho_bg);
%       p.addParameter('alpha_bg',      obj.alpha_bg);
%       p.addParameter('alpha_power',   obj.alpha_power);
%       p.addParameter('dc_mean',       obj.dc_mean);
%       p.addParameter('dr_mean',       obj.dr_mean);
%       p.addParameter('sigma_c',       obj.sigma_c);
%       p.addParameter('sigma_r',       obj.sigma_r);
%       p.addParameter('alpha_inside_scale', obj.alpha_inside_scale);
%       p.addParameter('c_cooper',      obj.c_cooper);
%       p.addParameter('rho_cooper',    obj.rho_cooper);
%       p.addParameter('alpha_cooper',  obj.alpha_cooper);
%       p.addParameter('cooper_sigma_c',obj.cooper_sigma_c);
%       p.addParameter('cooper_sigma_r',obj.cooper_sigma_r);
%       p.addParameter('rngSeed',       []);
%       p.parse(varargin{:});
%       S = p.Results;
% 
%       % apply overrides
%       fns = fieldnames(S);
%       for i = 1:numel(fns), obj.(fns{i}) = S.(fns{i}); end
% 
%       % --------- optional seeding (reproducible speckle) ----------
%       if ~isempty(obj.rngSeed), rng(obj.rngSeed); end
% 
%       % --------- short-hands ----------
%       Nx = arrayObj.Nx; Ny = arrayObj.Ny;
%       dx = arrayObj.dx;                  %#ok<NASGU>  (available if you want px↔mm)
%       y_vec = arrayObj.kgrid.y_vec;
%       x_src = arrayObj.x_src;
% 
%       %% 1) Place inclusion into grid (obj.inc_mask)
%       Ms = obj.inclusionMask;
%       Hm = size(Ms,1) * arrayObj.dx;
%       cx_from_top = (obj.z_mask_top_mm/1e3) + Hm/2;
%       ix_t = x_src + round(cx_from_top / arrayObj.dx);
%       iy_t = round(interp1(y_vec, 1:Ny, obj.y_target_mm/1e3, 'nearest', 'extrap'));
% 
%       hx = floor(size(Ms,1)/2);  hy = floor(size(Ms,2)/2);
%       ix_t = min(max(ix_t, 1 + hx), Nx - hx);
%       iy_t = min(max(iy_t, 1 + hy), Ny - hy);
% 
%       top  = ix_t - hx; left = iy_t - hy;
%       rIdx = max(1, top)  : min(Nx, top  + size(Ms,1) - 1);
%       cIdx = max(1, left) : min(Ny, left + size(Ms,2) - 1);
%       srcR = (rIdx - top + 1);
%       srcC = (cIdx - left + 1);
% 
%       obj.inc_mask = false(Nx,Ny);
%       obj.inc_mask(rIdx, cIdx) = Ms(srcR, srcC);
% 
%       obj.coop_mask = obj.cooperMask;  % already grid-sized
% 
%       %% 2) One speckle realization (uniform σ), then add MEAN inside
%       g = fspecial('gaussian', [7 7], 1.0);
%       n_r = conv2(randn(Nx,Ny,'single'), g, 'same');
%       n_c = conv2(randn(Nx,Ny,'single'), g, 'same');
% 
%       zn  = @(A) single( (A - mean(A(:)))./std(A(:)) );
%       n_r = zn(n_r);   n_c = zn(n_c);
% 
%       rho = single(obj.rho_bg) .* (1 + obj.sigma_r * n_r);
%       c   = single(obj.c_bg)   .* (1 + obj.sigma_c * n_c);
% 
%       % add mean contrasts in inclusion (pop without big variance change)
%       rho(obj.inc_mask) = rho(obj.inc_mask) .* (1 + obj.dr_mean);
%       c(obj.inc_mask)   = c(obj.inc_mask)   .* (1 + obj.dc_mean);
% 
%       % tighter clipping (stability & reduced amplitude swings)
%       rho = min(max(rho, single(obj.rho_bg*(1-0.05))), single(obj.rho_bg*(1+0.05)));
%       c   = min(max(c,   single(obj.c_bg  *(1-0.02))), single(obj.c_bg  *(1+0.02)));
% 
%       % attenuation (background with slightly higher inside)
%       alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
%       alpha(obj.inc_mask) = single(obj.alpha_inside_scale * obj.alpha_bg);
% 
%       % Cooper's overlay with tiny texture
%       if any(obj.coop_mask(:))
%         idx = obj.coop_mask;
%         % small texture to avoid flat patches
%         n_rC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%         n_cC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%         c(idx)   = single(obj.c_cooper)   .* (1 + obj.cooper_sigma_c * n_cC(idx));
%         rho(idx) = single(obj.rho_cooper) .* (1 + obj.cooper_sigma_r * n_rC(idx));
%         alpha(idx) = single(obj.alpha_cooper);
%       end
% 
%       % store baseline
%       obj.sound_speed = c;
%       obj.density     = rho;
%       obj.alpha_coeff = alpha;
% 
%       %% 3) Deform the MAPS (not the phantom) using your logic
%       % Convert displacements to pixel shifts using your ConvertMMToPX + scaling
%       Uax = ConvertMMToPX(obj.simresult.axial_disp)*-1;    % rows/depth
%       Ulat= ConvertMMToPX(obj.simresult.lateral_disp)*-1;  % cols/lateral
% 
%       [NR,NC] = size(obj.inc_mask);
%       if ~isequal(size(Uax),[NR,NC]), Uax  = imresize(Uax, [NR,NC]); end
%       if ~isequal(size(Ulat),[NR,NC]), Ulat = imresize(Ulat,[NR,NC]); end
% 
%       % NOTE: follow your current scale (×10). If you later calibrate properly,
%       % replace 10 with (1/(arrayObj.dx*1e3)) etc.
%       dR = Uax * 10;
%       dC = Ulat * 10;
% 
%       [Cs,Rs] = meshgrid(1:NC, 1:NR);
%       Rsrc = Rs - dR;  Csrc = Cs - dC;
%       Rsrc = max(1, min(NR, Rsrc));
%       Csrc = max(1, min(NC, Csrc));
% 
%       obj.inc_mask_def  = interp2(Cs, Rs, double(obj.inc_mask),  Csrc, Rsrc, 'cubic', 0) > 0.5;
%       obj.coop_mask_def = interp2(Cs, Rs, double(obj.coop_mask), Csrc, Rsrc, 'cubic', 0) > 0.5;
%       obj.inc_mask_def  = imfill(obj.inc_mask_def, 'holes');  obj.inc_mask_def  = bwareaopen(obj.inc_mask_def, 8);
%       obj.coop_mask_def = imfill(obj.coop_mask_def,'holes');  obj.coop_mask_def = bwareaopen(obj.coop_mask_def, 8);
% 
%       % Backward-warp the maps (c, rho, alpha) with cubic
%       obj.sound_speed_def  = single(interp2(Cs, Rs, double(obj.sound_speed),  Csrc, Rsrc, 'cubic', 0));
%       obj.density_def      = single(interp2(Cs, Rs, double(obj.density),      Csrc, Rsrc, 'cubic', 0));
%       obj.alpha_coeff_def  = single(interp2(Cs, Rs, double(obj.alpha_coeff),  Csrc, Rsrc, 'cubic', 0));
%     end
% 
%     function [med0, med1] = getMediumStructs(obj)
%       % Baseline
%       med0 = struct();
%       med0.sound_speed = obj.sound_speed;
%       med0.density     = obj.density;
%       med0.alpha_power = obj.alpha_power;       % scalar
%       med0.alpha_coeff = obj.alpha_coeff;
% 
%       % Deformed
%       med1 = struct();
%       med1.sound_speed = obj.sound_speed_def;
%       med1.density     = obj.density_def;
%       med1.alpha_power = obj.alpha_power;       % scalar (same as baseline)
%       med1.alpha_coeff = obj.alpha_coeff_def;
%     end
%   end
% end

% classdef KWaveMedium
%   %KWaveMedium Build baseline AND deformed k-Wave medium maps.
%   % - Same speckle realization for pre/post (seedable)
%   % - Inclusion "pop": small mean offsets (rho +2%, c +3%) with uniform σ
%   % - Realistic attenuation; slightly higher inside (stiff inclusion)
%   % - Warps the *maps* (c, rho, alpha) using FEA displacements
%   % - Robust mm→px conversion, no coordinate clamping, benign edge handling
%   %
%   % Usage:
%   %   med = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, ...
%   %           'rngSeed', 42, 'DispUnits','mm', 'DispSign', -1, 'Debug', true);
%   %   [med0, med1] = med.getMediumStructs();   % baseline, deformed
% 
%   %% -------------------- immutable inputs --------------------
%   properties (SetAccess = immutable)
%     arrayObj
%     inclusionMask logical         % raw Ms before placement
%     cooperMask    logical         % Nx-by-Ny logical (true where Cooper's exists)
%     simresult                         % struct with axial_disp, lateral_disp (mm or px)
%   end
% 
%   % placement
%   properties (SetAccess = immutable)
%     z_mask_top_mm (1,1) double = 6
%     y_target_mm   (1,1) double = 0
%   end
% 
%   % background material defaults (physical-ish)
%   properties (SetAccess = immutable)
%     c_bg   (1,1) double = 1540
%     rho_bg (1,1) double = 1000
%     alpha_power (1,1) double = 1.5
%     alpha_bg    (1,1) double = 0.6     % dB/(MHz^alpha_power*cm)
%   end
% 
%   % inclusion contrasts (kept modest to preserve correlation)
%   properties (SetAccess = immutable)
%     dc_mean  (1,1) double = +0.03      % +3% c inside
%     dr_mean  (1,1) double = +0.02      % +2% rho inside
%     sigma_c  (1,1) double = 0.005      % 0.5% c everywhere
%     sigma_r  (1,1) double = 0.02       % 2%   rho everywhere
%     alpha_inside_scale (1,1) double = 1.25  % 25% higher α in stiff inclusion
%   end
% 
%   % Cooper's fixed properties + tiny texture
%   properties (SetAccess = immutable)
%     c_cooper   (1,1) double = 1650
%     rho_cooper (1,1) double = 1100
%     alpha_cooper (1,1) double = 0.7
%     cooper_sigma_c (1,1) double = 0.003
%     cooper_sigma_r (1,1) double = 0.01
%   end
% 
%   % RNG / reproducibility
%   properties (SetAccess = immutable)
%     rngSeed   % [] = no change; otherwise sets rng(seed) at construction
%   end
% 
%   % deformation options
%   properties (SetAccess = immutable)
%     DispUnits char = 'mm'       % 'mm' or 'px'
%     DispSign  double = +1       % multiply displacements by this (use -1 if your sign is inverted)
%     CapStrainPct double = 0.02  % cap ≈2% of image size (per-dimension)
%     CapShiftPx  double = 5      % absolute pixel cap (each direction)
%     Debug logical = false
%   end
% 
%   %% -------------------- outputs (baseline + deformed) --------------------
%   properties (SetAccess = private)
%     % placed binary masks
%     inc_mask   logical
%     coop_mask  logical
% 
%     % baseline maps
%     density      single
%     sound_speed  single
%     alpha_coeff  single
% 
%     % deformed maps
%     density_def     single
%     sound_speed_def single
%     alpha_coeff_def single
%     inc_mask_def    logical
%     coop_mask_def   logical
%   end
% 
%   methods
%     function obj = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, varargin)
%       % --------- store inputs ----------
%       obj.arrayObj      = arrayObj;
%       obj.inclusionMask = logical(Ms);
%       obj.simresult     = simresult;
% 
%       if isempty(cooper_mask)
%         obj.cooperMask = false(arrayObj.Nx, arrayObj.Ny);
%       else
%         cm = logical(cooper_mask);
%         if isequal(size(cm), [arrayObj.Nx, arrayObj.Ny])
%           obj.cooperMask = cm;
%         else
%           error('cooper_mask must be Nx-by-Ny to match the grid.');
%         end
%       end
% 
%       % --------- parse overrides ----------
%       p = inputParser;
%       p.addParameter('z_mask_top_mm', obj.z_mask_top_mm);
%       p.addParameter('y_target_mm',   obj.y_target_mm);
%       p.addParameter('c_bg',          obj.c_bg);
%       p.addParameter('rho_bg',        obj.rho_bg);
%       p.addParameter('alpha_bg',      obj.alpha_bg);
%       p.addParameter('alpha_power',   obj.alpha_power);
%       p.addParameter('dc_mean',       obj.dc_mean);
%       p.addParameter('dr_mean',       obj.dr_mean);
%       p.addParameter('sigma_c',       obj.sigma_c);
%       p.addParameter('sigma_r',       obj.sigma_r);
%       p.addParameter('alpha_inside_scale', obj.alpha_inside_scale);
%       p.addParameter('c_cooper',      obj.c_cooper);
%       p.addParameter('rho_cooper',    obj.rho_cooper);
%       p.addParameter('alpha_cooper',  obj.alpha_cooper);
%       p.addParameter('cooper_sigma_c',obj.cooper_sigma_c);
%       p.addParameter('cooper_sigma_r',obj.cooper_sigma_r);
%       p.addParameter('rngSeed',       []);
%       p.addParameter('DispUnits',     obj.DispUnits);
%       p.addParameter('DispSign',      obj.DispSign);
%       p.addParameter('CapStrainPct',  obj.CapStrainPct);
%       p.addParameter('CapShiftPx',    obj.CapShiftPx);
%       p.addParameter('Debug',         obj.Debug);
%       p.parse(varargin{:});
%       S = p.Results;
% 
%       % apply overrides
%       fns = fieldnames(S);
%       for i = 1:numel(fns), obj.(fns{i}) = S.(fns{i}); end
% 
%       % --------- optional seeding (reproducible speckle) ----------
%       if ~isempty(obj.rngSeed), rng(obj.rngSeed); end
% 
%       % --------- short-hands ----------
%       Nx = arrayObj.Nx; Ny = arrayObj.Ny;
%       y_vec = arrayObj.kgrid.y_vec;
%       x_src = arrayObj.x_src;
% 
%       %% 1) Place inclusion into grid (obj.inc_mask)
%       Ms = obj.inclusionMask;
%       Hm = size(Ms,1) * arrayObj.dx;
%       cx_from_top = (obj.z_mask_top_mm/1e3) + Hm/2;
%       ix_t = x_src + round(cx_from_top / arrayObj.dx);
%       iy_t = round(interp1(y_vec, 1:Ny, obj.y_target_mm/1e3, 'nearest', 'extrap'));
% 
%       hx = floor(size(Ms,1)/2);  hy = floor(size(Ms,2)/2);
%       ix_t = min(max(ix_t, 1 + hx), Nx - hx);
%       iy_t = min(max(iy_t, 1 + hy), Ny - hy);
% 
%       top  = ix_t - hx; left = iy_t - hy;
%       rIdx = max(1, top)  : min(Nx, top  + size(Ms,1) - 1);
%       cIdx = max(1, left) : min(Ny, left + size(Ms,2) - 1);
%       srcR = (rIdx - top + 1);
%       srcC = (cIdx - left + 1);
% 
%       obj.inc_mask = false(Nx,Ny);
%       obj.inc_mask(rIdx, cIdx) = Ms(srcR, srcC);
%       obj.coop_mask = obj.cooperMask;  % already grid-sized
% 
%       %% 2) One speckle realization (uniform σ), then add MEAN inside
%       g = fspecial('gaussian', [7 7], 1.0);
%       n_r = conv2(randn(Nx,Ny,'single'), g, 'same');
%       n_c = conv2(randn(Nx,Ny,'single'), g, 'same');
% 
%       zn  = @(A) single( (A - mean(A(:)))./std(A(:)) );
%       n_r = zn(n_r);   n_c = zn(n_c);
% 
%       rho = single(obj.rho_bg) .* (1 + obj.sigma_r * n_r);
%       c   = single(obj.c_bg)   .* (1 + obj.sigma_c * n_c);
% 
%       % add mean contrasts in inclusion (pop without big variance change)
%       rho(obj.inc_mask) = rho(obj.inc_mask) .* (1 + obj.dr_mean);
%       c(obj.inc_mask)   = c(obj.inc_mask)   .* (1 + obj.dc_mean);
% 
%       % tighter clipping (stability & reduced amplitude swings)
%       rho = min(max(rho, single(obj.rho_bg*(1-0.05))), single(obj.rho_bg*(1+0.05)));
%       c   = min(max(c,   single(obj.c_bg  *(1-0.02))), single(obj.c_bg  *(1+0.02)));
% 
%       % attenuation (background with slightly higher inside)
%       alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
%       alpha(obj.inc_mask) = single(obj.alpha_inside_scale * obj.alpha_bg);
% 
%       % Cooper's overlay with tiny texture
%       if any(obj.coop_mask(:))
%         idx = obj.coop_mask;
%         n_rC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%         n_cC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%         c(idx)   = single(obj.c_cooper)   .* (1 + obj.cooper_sigma_c * n_cC(idx));
%         rho(idx) = single(obj.rho_cooper) .* (1 + obj.cooper_sigma_r * n_rC(idx));
%         alpha(idx) = single(obj.alpha_cooper);
%       end
% 
%       % store baseline
%       obj.sound_speed = c;
%       obj.density     = rho;
%       obj.alpha_coeff = alpha;
% 
%       %% 3) Deform the MAPS (not the phantom)
%       % fetch displacements and sign
%       Uax  = obj.DispSign * obj.simresult.axial_disp;    % rows/depth
%       Ulat = obj.DispSign * obj.simresult.lateral_disp;  % cols/lateral
% 
%       % resize to grid if needed
%       [NR,NC] = size(obj.inc_mask);
%       if ~isequal(size(Uax), [NR,NC]),  Uax  = imresize(Uax,  [NR,NC]); end
%       if ~isequal(size(Ulat),[NR,NC]),  Ulat = imresize(Ulat, [NR,NC]); end
% 
%       % mm -> px (or passthrough if already px)
%       if strcmpi(obj.DispUnits,'mm')
%         dx_mm = obj.arrayObj.dx * 1e3;
%         dy_mm = obj.arrayObj.dy * 1e3;
%         dR = Uax  ./ dx_mm;
%         dC = Ulat ./ dy_mm;
%       else
%         dR = Uax;
%         dC = Ulat;
%       end
% 
%       % cap shifts to realistic range (≈2% or ≤5 px by default)
%       if isfinite(obj.CapStrainPct) || isfinite(obj.CapShiftPx)
%         max_ax = min(obj.CapShiftPx, obj.CapStrainPct * NR);
%         max_lat= min(obj.CapShiftPx, obj.CapStrainPct * NC);
%         dR = max(min(dR,  max_ax), -max_ax);
%         dC = max(min(dC,  max_lat), -max_lat);
%       end
% 
%       % backward warp coords (NO CLAMPING)
%       [Cs,Rs] = meshgrid(1:NC, 1:NR);
%       Rsrc = Rs - dR;
%       Csrc = Cs - dC;
% 
%       % masks: interpolate then threshold (linear avoids ringing)
%       F_mi = griddedInterpolant(Rs, Cs, double(obj.inc_mask),  'linear', 'nearest');
%       F_mc = griddedInterpolant(Rs, Cs, double(obj.coop_mask), 'linear', 'nearest');
%       obj.inc_mask_def  = F_mi(Rsrc, Csrc) > 0.5;
%       obj.coop_mask_def = F_mc(Rsrc, Csrc) > 0.5;
%       obj.inc_mask_def  = imfill(obj.inc_mask_def, 'holes');  obj.inc_mask_def  = bwareaopen(obj.inc_mask_def, 8);
%       obj.coop_mask_def = imfill(obj.coop_mask_def,'holes');  obj.coop_mask_def = bwareaopen(obj.coop_mask_def, 8);
% 
%       % fields: cubic interpolation, 'nearest' extrap to avoid long bands
%       F_c   = griddedInterpolant(Rs, Cs, double(obj.sound_speed), 'cubic', 'nearest');
%       F_rho = griddedInterpolant(Rs, Cs, double(obj.density),     'cubic', 'nearest');
%       F_alp = griddedInterpolant(Rs, Cs, double(obj.alpha_coeff), 'cubic', 'nearest');
% 
%       obj.sound_speed_def = single(F_c(Rsrc, Csrc));
%       obj.density_def     = single(F_rho(Rsrc, Csrc));
%       obj.alpha_coeff_def = single(F_alp(Rsrc, Csrc));
% 
% 
%       % optional debug: report shifts & correlations
%       if obj.Debug
%         fprintf('ax px: min %.2f  med %.2f  max %.2f\n', min(dR,[],'all'), median(dR,'all'), max(dR,[],'all'));
%         fprintf('lt px: min %.2f  med %.2f  max %.2f\n', min(dC,[],'all'), median(dC,'all'), max(dC,[],'all'));
%         r1 = corrcoef(double(obj.density(:)),     double(obj.density_def(:)));     fprintf('corr(rho)   = %.4f\n', r1(1,2));
%         r2 = corrcoef(double(obj.sound_speed(:)), double(obj.sound_speed_def(:))); fprintf('corr(c)     = %.4f\n', r2(1,2));
%         r3 = corrcoef(double(obj.alpha_coeff(:)), double(obj.alpha_coeff_def(:))); fprintf('corr(alpha) = %.4f\n', r3(1,2));
% 
%         % Build the same grids and shifts you used when creating the deformed maps:
%         [NR,NC] = size(obj.sound_speed);
%         [Cs,Rs] = meshgrid(1:NC, 1:NR);
%         % dR, dC are your px shifts (with units + sign handled already)
%         Rsrc = Rs - dR; Csrc = Cs - dC;
% 
%         % Forward-warp the baseline (this reproduces your *_def maps)
%         Fc = griddedInterpolant(Rs, Cs, double(obj.sound_speed), 'cubic', 'nearest');
%         c_def_pred = single(Fc(Rsrc, Csrc));
% 
%         % Compare to the stored deformed map (should be ~1.0)
%         r_c = corrcoef(double(c_def_pred(:)), double(obj.sound_speed_def(:)));
%         disp(['compensated corr(c) = ' num2str(r_c(1,2))]);
% 
%         % Do the same for rho if you like
%         Fr = griddedInterpolant(Rs, Cs, double(obj.density), 'cubic', 'nearest');
%         rho_def_pred = single(Fr(Rsrc, Csrc));
%         r_rho = corrcoef(double(rho_def_pred(:)), double(obj.density_def(:)));
%         disp(['compensated corr(rho) = ' num2str(r_rho(1,2))]);
%       end
%     end
% 
%     function [med0, med1] = getMediumStructs(obj)
%       % Baseline
%       med0 = struct();
%       med0.sound_speed = obj.sound_speed;
%       med0.density     = obj.density;
%       med0.alpha_power = obj.alpha_power;       % scalar
%       med0.alpha_coeff = obj.alpha_coeff;
% 
%       % Deformed
%       med1 = struct();
%       med1.sound_speed = obj.sound_speed_def;
%       med1.density     = obj.density_def;
%       med1.alpha_power = obj.alpha_power;       % scalar (same as baseline)
%       med1.alpha_coeff = obj.alpha_coeff_def;
%     end
%   end
% end


% classdef KWaveMedium
%   %KWaveMedium Build baseline AND deformed k-Wave medium maps.
%   % - Reuses the SAME speckle realization for pre/post
%   % - Adds mean contrast in the inclusion (rho +2%, c +3%)
%   % - Keeps variance small and uniform to reduce decorrelation
%   % - Higher attenuation in stiff inclusion
%   % - Optionally adds tiny texture to Cooper's ligaments
%   %
%   % Usage:
%   %   med = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, ...
%   %         'z_mask_top_mm', 6, 'y_target_mm', 0, ...
%   %         'rngSeed', 42);
%   %
%   %   [med0, med1] = med.getMediumStructs();   % baseline, deformed
% 
% 
%   %% -------------------- immutable inputs --------------------
%   properties (SetAccess = immutable)
%     arrayObj
%     inclusionMask logical         % raw Ms before placement
%     cooperMask    logical         % Nx-by-Ny logical (true where Cooper's exists)
%     simresult                         % struct with axial_disp, lateral_disp (mm or pre-converted)
%   end
% 
%   % placement
%   properties (SetAccess = immutable)
%     z_mask_top_mm (1,1) double = 6
%     y_target_mm   (1,1) double = 0
%   end
% 
%   % % background material defaults (physical-ish)
%   % properties (SetAccess = immutable)
%   %   c_bg   (1,1) double = 1540
%   %   rho_bg (1,1) double = 1000
%   %   alpha_power (1,1) double = 1.5
%   %   alpha_bg    (1,1) double = 0.6     % dB/(MHz^alpha_power*cm)
%   % end
%   % 
%   % % inclusion contrasts (kept modest to preserve correlation)
%   % properties (SetAccess = immutable)
%   %   dc_mean  (1,1) double = +0.03      % +3% c inside
%   %   dr_mean  (1,1) double = +0.02      % +2% rho inside
%   %   sigma_c  (1,1) double = 0.005      % 0.5% c everywhere
%   %   sigma_r  (1,1) double = 0.02       % 2%   rho everywhere
%   %   alpha_inside_scale (1,1) double = 1.25  % 25% higher α in stiff inclusion
%   % end
%   % 
%   % % Cooper's fixed properties + tiny texture
%   % properties (SetAccess = immutable)
%   %   c_cooper   (1,1) double = 1650
%   %   rho_cooper (1,1) double = 1100
%   %   alpha_cooper (1,1) double = 0.7    % usually high in fibrous tissue
%   %   cooper_sigma_c (1,1) double = 0.003
%   %   cooper_sigma_r (1,1) double = 0.01
%   % end
% 
%   properties (SetAccess = immutable)
%     % background
%       c_bg   (1,1) double = 1540
%     rho_bg (1,1) double = 1040
%     alpha_power (1,1) double = 1.1       % linear-with-frequency for simplicity
%     alpha_bg    (1,1) double = 0.75       % dB/(MHz·cm) light attenuation
% 
%     % inclusion contrasts
%     dc_mean  (1,1) double = +0.06        % ~+6% c inside (modest stiffness)
%     dr_mean  (1,1) double = 0.05         % NO density contrast
%     sigma_c  (1,1) double = 0.003        % 0.2% c texture (very small)
%     sigma_r  (1,1) double = 0.01         % NO density texture
%     alpha_inside_scale (1,1) double = 1.5% NO α contrast
% 
%     % Cooper's ligament
%     c_cooper   (1,1) double = 1600       % ~+3.9% over bg (modest)
%     rho_cooper (1,1) double = 1000       % same as bg (no density contrast)
%     alpha_cooper (1,1) double = 0.3      % same as bg (no α contrast)
%     cooper_sigma_c (1,1) double = 0.001  % tiny texture
%     cooper_sigma_r (1,1) double = 0.00   % no density texture
% 
%   end
% 
%   % RNG / reproducibility
%   properties (SetAccess = immutable)
%     rngSeed   % [] = no change; otherwise sets rng(seed) at construction
%   end
% 
%   %% -------------------- outputs (baseline + deformed) --------------------
%   properties (SetAccess = private)
%     % placed binary masks
%     inc_mask   logical
%     coop_mask  logical
% 
%     % baseline maps
%     density      single
%     sound_speed  single
%     alpha_coeff  single
% 
%     % deformed maps
%     density_def     single
%     sound_speed_def single
%     alpha_coeff_def single
%     inc_mask_def    logical
%     coop_mask_def   logical
%   end
% 
%   methods
%     function obj = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, varargin)
%       % --------- store inputs ----------
%       obj.arrayObj      = arrayObj;
%       obj.inclusionMask = logical(Ms);
%       obj.simresult     = simresult;
% 
%       if isempty(cooper_mask)
%         obj.cooperMask = false(arrayObj.Nx, arrayObj.Ny);
%       else
%         cm = logical(cooper_mask);
%         if isequal(size(cm), [arrayObj.Nx, arrayObj.Ny])
%           obj.cooperMask = cm;
%         else
%           error('cooper_mask must be Nx-by-Ny to match the grid.');
%         end
%       end
% 
%       % --------- parse overrides ----------
%       p = inputParser;
%       p.addParameter('z_mask_top_mm', obj.z_mask_top_mm);
%       p.addParameter('y_target_mm',   obj.y_target_mm);
%       p.addParameter('c_bg',          obj.c_bg);
%       p.addParameter('rho_bg',        obj.rho_bg);
%       p.addParameter('alpha_bg',      obj.alpha_bg);
%       p.addParameter('alpha_power',   obj.alpha_power);
%       p.addParameter('dc_mean',       obj.dc_mean);
%       p.addParameter('dr_mean',       obj.dr_mean);
%       p.addParameter('sigma_c',       obj.sigma_c);
%       p.addParameter('sigma_r',       obj.sigma_r);
%       p.addParameter('alpha_inside_scale', obj.alpha_inside_scale);
%       p.addParameter('c_cooper',      obj.c_cooper);
%       p.addParameter('rho_cooper',    obj.rho_cooper);
%       p.addParameter('alpha_cooper',  obj.alpha_cooper);
%       p.addParameter('cooper_sigma_c',obj.cooper_sigma_c);
%       p.addParameter('cooper_sigma_r',obj.cooper_sigma_r);
%       p.addParameter('rngSeed',       []);
%       % p.addParameter('DispSign',1);
%       p.parse(varargin{:});
%       S = p.Results;
% 
%       % apply overrides
%       fns = fieldnames(S);
%       for i = 1:numel(fns), obj.(fns{i}) = S.(fns{i}); end
% 
%       % --------- optional seeding (reproducible speckle) ----------
%       if ~isempty(obj.rngSeed), rng(obj.rngSeed); end
% 
%       % --------- short-hands ----------
%       Nx = arrayObj.Nx; Ny = arrayObj.Ny;
%       y_vec = arrayObj.kgrid.y_vec;
%       x_src = arrayObj.x_src;
% 
%       %% 1) Place inclusion into grid (obj.inc_mask)
%       Ms = obj.inclusionMask;
%       % Hm = size(Ms,1) * arrayObj.dx;
%       % cx_from_top = (obj.z_mask_top_mm/1e3) + Hm/2;
%       % ix_t = x_src + round(cx_from_top / arrayObj.dx);
%       Hm = size(Ms,1) * arrayObj.dx;                       % mask physical height
%         cx_from_top = (obj.z_mask_top_mm/1e3) + Hm/2;        % center depth in m
%         ix_t = round(cx_from_top / arrayObj.dx);             % index from top of grid
% 
%       iy_t = round(interp1(y_vec, 1:Ny, obj.y_target_mm/1e3, 'nearest', 'extrap'));
% 
%       hx = floor(size(Ms,1)/2);  hy = floor(size(Ms,2)/2);
%       ix_t = min(max(ix_t, 1 + hx), Nx - hx);
%       iy_t = min(max(iy_t, 1 + hy), Ny - hy);
% 
%       top  = ix_t - hx; left = iy_t - hy;
%       rIdx = max(1, top)  : min(Nx, top  + size(Ms,1) - 1);
%       cIdx = max(1, left) : min(Ny, left + size(Ms,2) - 1);
%       srcR = (rIdx - top + 1);
%       srcC = (cIdx - left + 1);
% 
%       obj.inc_mask = false(Nx,Ny);
%       obj.inc_mask(rIdx, cIdx) = Ms(srcR, srcC);
% 
%       obj.coop_mask = obj.cooperMask;  % already grid-sized
% 
%       %% 2) One speckle realization (uniform σ), then add MEAN inside
%       % g  = fspecial('gaussian', [7 7], 1.0);
% % Given your array object:
% dx   = arrayObj.dx;           % grid step (m)
% c0   = arrayObj.c0;
% f0   = arrayObj.f0;
% lambda = c0/f0;          % wavelength (m)
% 
% corr_in_lambda = 1.0;    % try 1.0 λ (0.5–1.5 are reasonable)
% FWHM_m   = corr_in_lambda * lambda;
% sigma_px = (FWHM_m / (2.355*dx));          % convert to pixel sigma
% ksz      = 2*ceil(3*sigma_px)+1;           % >= 6σ, force odd size
% 
% g = fspecial('gaussian', [ksz ksz], sigma_px);
% 
%       n0  = conv2(randn(Nx,Ny,'single'), g, 'same');
%         zn = @(A) single( (A - mean(A(:)))./std(A(:)+eps) );
%         n0  = zn(n0);
% 
%         rho = single(obj.rho_bg) .* (1 + obj.sigma_r * n0);
%         c   = single(obj.c_bg)   .* (1 + obj.sigma_c * n0);
% 
% 
%       % (a) clip only the noise field BEFORE adding means
%         rho = min(max(rho, obj.rho_bg*(1-obj.sigma_r)), obj.rho_bg*(1+obj.sigma_r));
%         c   = min(max(c,   obj.c_bg  *(1-obj.sigma_c)), obj.c_bg  *(1+obj.sigma_c));
%         % (b) then add the inclusion means (will not be clipped)
%         % rho(obj.inc_mask) = rho(obj.inc_mask) .* (1 + obj.dr_mean);
%         % c(obj.inc_mask)   = c(obj.inc_mask)   .* (1 + obj.dc_mean);
%         % c(obj.inc_mask)   = obj.c_bg * (1 + obj.dc_mean);           
%         % rho(obj.inc_mask) = obj.rho_bg * (1 + obj.dr_mean);
%         % inside_sigma = 0.005;        % much smoother (hypoechoic)
%         % n_c = conv2(randn(Nx,Ny,'single'), g, 'same');
%         % n_c = (n_c - mean(n_c(:))) ./ std(n_c(:));
%         % c   = single(obj.c_bg) .* (1 + obj.sigma_c * n_c);
% 
%          % c(obj.inc_mask) = single(obj.c_bg) .* (1 + 0.02 * n(obj.inc_mask)) .* (1 + obj.dc_mean);
%         % rho(obj.inc_mask) = single(obj.c_bg) .* (1+ 0.015 * n(obj.inc_mask)).* (1 + obj.dr_mean);
% 
%        % figure;imagesc(obj.inc_mask);
% 
%        % --- inclusion location/size (in pixels) ---
%     cy = round(0.45*Nx);     % depth row
%     cx = round(0.60*Ny);     % lateral col
%     r_px = round(0.10*min(Nx,Ny));
% 
%     % --- smooth circular mask ---
%     [X,Y] = ndgrid(1:Nx,1:Ny);
%     M     = ( (X-cy).^2 + (Y-cx).^2 ) <= r_px^2;
%     edge_px = 4;
%     D  = bwdist(M) - bwdist(~M);
%     Ms = 0.5 + 0.5*cos(pi*min(max(D/edge_px,-1),1));  % 0..1 smooth
%     Ms(D>= edge_px) = 1; Ms(D<=-edge_px) = 0;
% 
%     % --- STRONG mean contrast (start bold; dial back later) ---
%     Dc   = +0.03;    % +3% speed
%     Drho = +0.02;    % +2% density
% 
%     % apply mean offsets
%     c   = c   .* (1 + Dc*Ms);
%     rho = rho .* (1 + Drho*Ms);
% 
%     % --- reduce variance inside so it looks coherent ---
%     k_var_c   = 0.4;     % 40% of bg variance
%     k_var_rho = 0.5;
% 
%     g  = fspecial('gaussian',[7 7],1.0);
%     zn = @(A) single((A-mean(A(:)))./(std(A(:))+eps));
%     n_inc = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));     % same correlation
%     c   = c   + single(c0)*(k_var_c  *obj.sigma_c).* n_inc .* Ms;
%     rho = rho + single(obj.rho_bg)*(k_var_rho*obj.sigma_r).* n_inc .* Ms;
% 
% z_max_img = arrayObj.c0 * (arrayObj.Nt*arrayObj.dt) / 2;
% fprintf('Depth window: %.2f mm\n', 1e3*z_max_img);
% 
% 
%       % attenuation (background with slightly higher inside)
%       alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
%       alpha(obj.inc_mask) = single(obj.alpha_inside_scale * obj.alpha_bg);
% 
%       % Cooper's overlay with tiny texture
%       % if any(obj.coop_mask(:))
%       %   idx = obj.coop_mask;
%       %   n_rC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%       %   n_cC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
%       %   c(idx)   = single(obj.c_cooper)   .* (1 + obj.cooper_sigma_c * n_cC(idx));
%       %   rho(idx) = single(obj.rho_cooper) .* (1 + obj.cooper_sigma_r * n_rC(idx));
%       %   % alpha(idx) = single(obj.alpha_cooper);
%       % end
% 
%     % % ---------- UNIFORM DENSITY ----------
%     % rho = single(obj.rho_bg) * ones(Nx,Ny,'single');  % no ρ contrast anywhere
%     % 
%     % % ---------- SOUND SPEED: small global texture + modest inclusion mean ----------
%     % g = fspecial('gaussian', [7 7], 1.0);
%     % n_c = conv2(randn(Nx,Ny,'single'), g, 'same'); 
%     % n_c = single( (n_c - mean(n_c(:)))./std(n_c(:)) );
%     % c   = single(obj.c_bg) .* (1 + obj.sigma_c * n_c);              % tiny texture
%     % c(obj.inc_mask) = single(obj.c_bg * (1 + obj.dc_mean));         % inclusion stiffer
%     % 
%     % % Cooper's ligament (sound speed only; modest)
%     % if any(obj.coop_mask(:))
%     %   n_cC = conv2(randn(Nx,Ny,'single'), g, 'same');
%     %   n_cC = single( (n_cC - mean(n_cC(:)))./std(n_cC(:)) );
%     %   idx = obj.coop_mask;
%     %   c(idx) = single(obj.c_cooper) .* (1 + obj.cooper_sigma_c * n_cC(idx));
%     % end
%     % 
%     % % (Optional) mild clipping only on c to avoid extremes in k-Wave
%     % c = min(max(c, single(obj.c_bg*(1-0.03))), single(obj.c_bg*(1+0.08)));
% 
%     % ---------- ATTENUATION: UNIFORM ----------
%     %alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
% 
% 
%       % store baseline
%       obj.sound_speed = c;
%       obj.density     = rho;
%       obj.alpha_coeff = alpha;
% 
%       %% 3) Deform the MAPS (not the phantom) — using griddedInterpolant
%       % (keeps identical speckle; only positions change)
%       % Uax  = ConvertMMToPX(obj.simresult.axial_disp)   * -1;  % rows/depth
%       % Ulat = ConvertMMToPX(obj.simresult.lateral_disp) * -1;  % cols/lateral
% 
%       % mm -> grid pixels (rows/cols)
%       % Uax = (-0.1) * obj.simresult.axial_disp   ./ (obj.arrayObj.dx * 1e3);  % axial [px]
%       % Ulat = (0) * obj.simresult.lateral_disp ./ (obj.arrayObj.dy * 1e3);  % lateral [px]
% 
%       % target inter-frame axial delta: 0.1 mm (≈0.3 λ at 5 MHz)
%         lambda_mm = (obj.c_bg/obj.arrayObj.f0) * 1e3;     % mm
%         target_mm = min(0.5*lambda_mm, 0.10);             % <= 0.5λ, cap at 0.10 mm
% 
%         peak_mm   = max(1e-6, max(abs(obj.simresult.axial_disp(:)))); % mm
%         scale_ax  = target_mm / peak_mm;
% 
%         fs = arrayObj.fs;           % Hz
%         c0 = obj.c_bg;              % m/s
%         lambda = c0/arrayObj.f0;    % m
%         px_per_sample = c0*arrayObj.dt/2 / arrayObj.dx;   % samples->pixels (2-way)
% 
%         fprintf('[MED] target_mm=%.3f (%.2f λ)\n', target_mm, target_mm/ (lambda*1e3));
% 
% 
%         % disp_ax = ConvertMMToPX(obj.simresult.axial_disp);
%         % disp_lat = ConvertMMToPX(obj.simresult.lateral_disp);
%         disp_ax = obj.simresult.axial_disp;
%         disp_lat = obj.simresult.lateral_disp;
% 
%         Uax  = (-scale_ax) * disp_ax./ (obj.arrayObj.dx * 1e3); % px
%         Ulat = (-scale_ax)         * disp_lat./ (obj.arrayObj.dy * 1e3); % px
%         assert(abs(max(Uax,[],'all')) < 2, 'Warp too large: >2 px will decorrelate small kernels');
%         fprintf('[MED] |Uax|max=%.3f px, 95%%=%.3f px\n', max(abs(Uax(:))), prctile(abs(Uax(:)),95));
% 
% 
% 
%       [NR,NC] = size(obj.inc_mask);
%       if ~isequal(size(Uax), [NR,NC]),  Uax  = imresize(Uax,  [NR,NC]); end
%       if ~isequal(size(Ulat),[NR,NC]),  Ulat = imresize(Ulat, [NR,NC]); end
% 
%       % NOTE: retaining your current scaling (*10). If you later calibrate,
%       % replace this with proper mm->px from arrayObj.dx/dy.
%       dR = Uax;   % axial shift in pixels
%       dC = Ulat;  % lateral shift in pixels
% 
%       % Backward warp coordinates (NO clamping)
%       [Cs,Rs] = meshgrid(1:NC, 1:NR);
%       Rsrc = Rs - dR;
%       Csrc = Cs - dC;
% 
%       oob = mean( Rsrc<1 | Rsrc>NR | Csrc<1 | Csrc>NC, 'all');
%       fprintf('OOB fraction = %.4f\n', oob);
% 
% 
%       % Masks: linear interpolation + 'nearest' extrap → then threshold
%       F_mi = griddedInterpolant(Rs, Cs, double(obj.inc_mask),  'linear', 'nearest');
%       F_mc = griddedInterpolant(Rs, Cs, double(obj.coop_mask), 'linear', 'nearest');
%       obj.inc_mask_def  = F_mi(Rsrc, Csrc) > 0.5;
%       obj.coop_mask_def = F_mc(Rsrc, Csrc) > 0.5;
%       obj.inc_mask_def  = imfill(obj.inc_mask_def, 'holes');  obj.inc_mask_def  = bwareaopen(obj.inc_mask_def, 8);
%       obj.coop_mask_def = imfill(obj.coop_mask_def,'holes');  obj.coop_mask_def = bwareaopen(obj.coop_mask_def, 8);
% 
%       % Fields: cubic interpolation + 'nearest' extrapolation
%       F_c   = griddedInterpolant(Rs, Cs, double(obj.sound_speed), 'linear', 'nearest');
%       F_rho = griddedInterpolant(Rs, Cs, double(obj.density),     'linear', 'nearest');
%       F_alp = griddedInterpolant(Rs, Cs, double(obj.alpha_coeff), 'linear', 'nearest');
% 
%       obj.sound_speed_def = single(F_c(Rsrc, Csrc));
%       obj.density_def     = single(F_rho(Rsrc, Csrc));
%       obj.alpha_coeff_def = single(F_alp(Rsrc, Csrc));
% 
% 
% % after you build the medium object:
% % med = KWaveMedium(...);
% % arr = your KWaveLinearArray
% 
% rows = find(any(obj.inc_mask, 2));              % rows where inclusion exists
% cols = find(any(obj.inc_mask, 1));              % cols where inclusion exists
% 
% z_mm = (rows - arrayObj.x_src) * arrayObj.dx * 1e3;       % convert from rows to mm (depth)
% y_mm = arrayObj.kgrid.y_vec(cols) * 1e3;             % lateral in mm
% 
% fprintf('Placed inclusion depth: [%.1f .. %.1f] mm\n', min(z_mm), max(z_mm));
% fprintf('Placed inclusion lateral: [%.1f .. %.1f] mm\n', min(y_mm), max(y_mm));
% 
% 
% 
%     end
% 
% 
%     function [med0, med1] = getMediumStructs(obj)
%       % Baseline
%       med0 = struct();
%       med0.sound_speed = obj.sound_speed;
%       med0.density     = obj.density;
%       med0.alpha_power = obj.alpha_power;       % scalar
%       med0.alpha_coeff = obj.alpha_coeff;
%       med0.alpha_mode  = 'no_dispersion';   
% 
%       % Deformed
%       med1 = struct();
%       med1.sound_speed = obj.sound_speed_def;
%       med1.density     = obj.density_def;
%       med1.alpha_power = obj.alpha_power;       % scalar (same as baseline)
%       med1.alpha_coeff = obj.alpha_coeff_def;
%       med1.alpha_mode = 'no_dispersion';
%     end
%   end
% end


classdef KWaveMedium
%KWaveMedium  Build baseline and deformed k-Wave medium maps (2D).
%
% Usage:
%   med = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, ...
%          'z_mask_top_mm', 6, 'y_target_mm', 0, 'rngSeed', 42);
%
%   [med0,med1] = med.getMediumStructs();   % baseline, deformed

  %% -------------------- immutable inputs --------------------
  properties (SetAccess = immutable)
    arrayObj
    inclusionMask logical            % binary mask (to be placed)
    cooperMask    logical            % Nx-by-Ny mask (optional)
    simresult                        % struct with axial_disp, lateral_disp in mm (optional)
  end

  % placement (center of Ms measured from top of grid and lateral axis)
  properties (SetAccess = immutable)
    z_mask_top_mm (1,1) double = 6
    y_target_mm   (1,1) double = 0
  end

  % background numbers + texture
  properties (SetAccess = immutable)
    c_bg   (1,1) double = 1540
    rho_bg (1,1) double = 1000
    alpha_power (1,1) double = 1.1
    alpha_bg    (1,1) double = 0.7            % dB/(MHz^alpha_power·cm)

    sigma_c_bg  (1,1) double = 0.02           % 2% std in background c
    sigma_r_bg  (1,1) double = 0.03           % 3% std in background ρ
    corr_in_lambda (1,1) double = 1.0         % speckle corr. length in λ
  end

  % inclusion mean contrasts + variance scaling
  properties (SetAccess = immutable)
    Dc_mean   (1,1) double = -0.03            % -3% c inside (hypoechoic)
    Drho_mean (1,1) double = -0.02            % -2% ρ inside
    k_var_c   (1,1) double = 0.5              % inside variance = k * bg var
    k_var_rho (1,1) double = 0.5
    alpha_inside_scale (1,1) double = 1.25    % 25% higher α inside
    smooth_edge_px (1,1) double = 4           % 0 => hard edge
  end

  % Cooper's ligament (optional)
  properties (SetAccess = immutable)
    c_cooper   (1,1) double = 1600
    rho_cooper (1,1) double = 1000
    alpha_cooper (1,1) double = 0.7
    cooper_sigma_c (1,1) double = 0.003
    cooper_sigma_r (1,1) double = 0.01
  end

  % RNG / reproducibility
  properties (SetAccess = immutable)
    rngSeed = []
  end

  %% -------------------- outputs --------------------
  properties (SetAccess = private)
    % placed masks
    inc_mask   logical
    coop_mask  logical

    % baseline fields
    density      single
    sound_speed  single
    alpha_coeff  single

    % deformed fields
    density_def     single
    sound_speed_def single
    alpha_coeff_def single
    inc_mask_def    logical
    coop_mask_def   logical
  end

  %% -------------------- methods --------------------
  methods
    function obj = KWaveMedium(arrayObj, Ms, cooper_mask, simresult, varargin)
      % Inputs
      obj.arrayObj      = arrayObj;
      obj.inclusionMask = logical(Ms);
      obj.simresult     = [];
      if exist('simresult','var') && ~isempty(simresult), obj.simresult = simresult; end

      if isempty(cooper_mask)
        obj.cooperMask = false(arrayObj.Nx, arrayObj.Ny);
      else
        cm = logical(cooper_mask);
        assert(isequal(size(cm), [arrayObj.Nx, arrayObj.Ny]), ...
          'cooper_mask must be Nx-by-Ny to match the grid.');
        obj.cooperMask = cm;
      end

      % Overrides
      p = inputParser;
      p.addParameter('z_mask_top_mm', obj.z_mask_top_mm);
      p.addParameter('y_target_mm',   obj.y_target_mm);
      p.addParameter('c_bg',          obj.c_bg);
      p.addParameter('rho_bg',        obj.rho_bg);
      p.addParameter('alpha_bg',      obj.alpha_bg);
      p.addParameter('alpha_power',   obj.alpha_power);
      p.addParameter('sigma_c_bg',    obj.sigma_c_bg);
      p.addParameter('sigma_r_bg',    obj.sigma_r_bg);
      p.addParameter('corr_in_lambda',obj.corr_in_lambda);
      p.addParameter('Dc_mean',       obj.Dc_mean);
      p.addParameter('Drho_mean',     obj.Drho_mean);
      p.addParameter('k_var_c',       obj.k_var_c);
      p.addParameter('k_var_rho',     obj.k_var_rho);
      p.addParameter('alpha_inside_scale', obj.alpha_inside_scale);
      p.addParameter('smooth_edge_px',obj.smooth_edge_px);
      p.addParameter('c_cooper',      obj.c_cooper);
      p.addParameter('rho_cooper',    obj.rho_cooper);
      p.addParameter('alpha_cooper',  obj.alpha_cooper);
      p.addParameter('cooper_sigma_c',obj.cooper_sigma_c);
      p.addParameter('cooper_sigma_r',obj.cooper_sigma_r);
      p.addParameter('rngSeed',       obj.rngSeed);
      p.parse(varargin{:});
      S = p.Results;
      fn = fieldnames(S); for k=1:numel(fn), obj.(fn{k}) = S.(fn{k}); end

      if ~isempty(obj.rngSeed), rng(obj.rngSeed); end

      Nx = arrayObj.Nx; Ny = arrayObj.Ny; dx = arrayObj.dx;
      y_vec = arrayObj.kgrid.y_vec;

      %% 1) Place inclusion mask on grid
      Ms_in  = obj.inclusionMask;
      Hm     = size(Ms_in,1) * dx;                      % mask height (m)
      cx_m   = (obj.z_mask_top_mm/1e3) + Hm/2;         % desired center depth
      ix_c   = round(cx_m / dx);                       % center row index
      iy_c   = round(interp1(y_vec, 1:Ny, obj.y_target_mm/1e3, 'nearest','extrap'));

      hx = floor(size(Ms_in,1)/2); hy = floor(size(Ms_in,2)/2);
      ix_c = min(max(ix_c, 1+hx), Nx-hx);
      iy_c = min(max(iy_c, 1+hy), Ny-hy);

      top  = ix_c - hx; left = iy_c - hy;
      rIdx = max(1, top)  : min(Nx, top  + size(Ms_in,1) - 1);
      cIdx = max(1, left) : min(Ny, left + size(Ms_in,2) - 1);
      srcR = (rIdx - top + 1); srcC = (cIdx - left + 1);

      inc_hard = false(Nx,Ny); inc_hard(rIdx, cIdx) = Ms_in(srcR, srcC);

      % optional smooth edge
      if obj.smooth_edge_px > 0
        D = bwdist(inc_hard) - bwdist(~inc_hard);
        Ms = 0.5 + 0.5*cos(pi*min(max(D/obj.smooth_edge_px, -1), 1));
        Ms(D >=  obj.smooth_edge_px)  = 1;
        Ms(D <= -obj.smooth_edge_px)  = 0;
      else
        Ms = single(inc_hard);
      end
      obj.inc_mask  = inc_hard;
      obj.coop_mask = obj.cooperMask;

      %% 2) Build ONE shared background speckle field (n0), then add inclusion means
      % correlation kernel size from wavelength
      lambda = obj.c_bg / arrayObj.f0;                        % m
      sigma_px = (obj.corr_in_lambda*lambda) / (2.355*dx);    % Gaussian σ in px
      ksz      = 2*ceil(3*sigma_px)+1;                        % >=6σ and odd
      g        = fspecial('gaussian', [ksz ksz], sigma_px);

      zn = @(A) single((A - mean(A(:))) ./ (std(A(:)) + eps));
      n0 = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));       % shared noise

      % background (NO renormalization after this!)
      c   = single(obj.c_bg)  .* (1 + obj.sigma_c_bg * n0);
      rho = single(obj.rho_bg).* (1 + obj.sigma_r_bg * n0);

      % inclusion mean offsets
      c   = c   .* (1 + obj.Dc_mean   * Ms);
      rho = rho .* (1 + obj.Drho_mean * Ms);

      % reduced variance inside (use another field with same corr. length)
      n_inc = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
      c   = c   + single(obj.c_bg)  * (obj.k_var_c  * obj.sigma_c_bg) .* n_inc .* Ms;
      rho = rho + single(obj.rho_bg)* (obj.k_var_rho* obj.sigma_r_bg) .* n_inc .* Ms;

      % attenuation
      alpha = single(obj.alpha_bg) * ones(Nx,Ny,'single');
      alpha = alpha .* (1 + (obj.alpha_inside_scale-1) * Ms);

      % Optional Cooper's overlay
      if any(obj.coop_mask(:))
        idx = obj.coop_mask;
        n_rC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
        n_cC = zn(conv2(randn(Nx,Ny,'single'), g, 'same'));
        c(idx)   = single(obj.c_cooper)   .* (1 + obj.cooper_sigma_c * n_cC(idx));
        rho(idx) = single(obj.rho_cooper) .* (1 + obj.cooper_sigma_r * n_rC(idx));
        alpha(idx) = single(obj.alpha_cooper);
      end

      % store baseline
      obj.sound_speed = c;
      obj.density     = rho;
      obj.alpha_coeff = alpha;

      %% 3) Deform fields (optional). If no simresult given → copy baseline.
      if isempty(obj.simresult)
        obj.sound_speed_def = obj.sound_speed;
        obj.density_def     = obj.density;
        obj.alpha_coeff_def = obj.alpha_coeff;
        obj.inc_mask_def    = obj.inc_mask;
        obj.coop_mask_def   = obj.coop_mask;
        return
      end

      % convert mm displacements to pixels (backward warp)
      Uax  = obj.simresult.axial_disp(2:end,2:end)  ./ (dx*1e3);     % rows
      Ulat = obj.simresult.lateral_disp(2:end,2:end)./ (dx*1e3);     % cols (dy==dx)
      
      dR = Uax;
      dC = Ulat;

% 

      % scale to safe magnitude if needed (≤ ~2 px to preserve correlation)
      mag95 = prctile(abs(Uax(:)),95);
      if mag95 > 2
        s = 2/mag95;
        Uax = Uax*s; Ulat = Ulat*s;
      end

      % [NR,NC] = size(obj.inc_mask);
      % if ~isequal(size(Uax),[NR,NC]),  Uax  = imresize(Uax,  [NR,NC]); end
      % if ~isequal(size(Ulat),[NR,NC]), Ulat = imresize(Ulat, [NR,NC]); end
      % 
      % [Cs,Rs] = meshgrid(1:NC,1:NR);
      % Rsrc = Rs - Uax;  Csrc = Cs - Ulat;
      % 
      % F = @(A) griddedInterpolant(Rs,Cs,double(A),'linear','nearest');
      % 
      % obj.sound_speed_def = single(F(obj.sound_speed)(Rsrc,Csrc));
      % obj.density_def     = single(F(obj.density)(Rsrc,Csrc));
      % obj.alpha_coeff_def = single(F(obj.alpha_coeff)(Rsrc,Csrc));
      % 
      % obj.inc_mask_def  = F(obj.inc_mask)(Rsrc,Csrc)  > 0.5;
      % obj.coop_mask_def = F(obj.coop_mask)(Rsrc,Csrc) > 0.5;
      % obj.inc_mask_def  = imfill(obj.inc_mask_def, 'holes');  obj.inc_mask_def  = bwareaopen(obj.inc_mask_def, 8);
      % obj.coop_mask_def = imfill(obj.coop_mask_def,'holes');  obj.coop_mask_def = bwareaopen(obj.coop_mask_def, 8);

      % --- build source grid once ---
    [NC, NR] = deal(size(obj.inc_mask,2), size(obj.inc_mask,1));   % or [NR,NC] = size(...)
    [Cs, Rs] = meshgrid(1:NC, 1:NR);
    
    % --- backward-warp sample locations you already computed ---
    Rsrc = Rs - dR;   Csrc = Cs - dC;   % (your code above)
    
    % ===== fields =====
    F_c   = griddedInterpolant(Rs, Cs, double(obj.sound_speed), 'linear','nearest');
    F_rho = griddedInterpolant(Rs, Cs, double(obj.density),     'linear','nearest');
    F_alp = griddedInterpolant(Rs, Cs, double(obj.alpha_coeff), 'linear','nearest');
    
    obj.sound_speed_def = single(F_c(Rsrc, Csrc));
    obj.density_def     = single(F_rho(Rsrc, Csrc));
    obj.alpha_coeff_def = single(F_alp(Rsrc, Csrc));
    
    % ===== masks (interpolate then threshold) =====
    F_mi = griddedInterpolant(Rs, Cs, double(obj.inc_mask),  'linear','nearest');
    F_mc = griddedInterpolant(Rs, Cs, double(obj.coop_mask), 'linear','nearest');
    
    obj.inc_mask_def  = F_mi(Rsrc, Csrc) > 0.5;
    obj.coop_mask_def = F_mc(Rsrc, Csrc) > 0.5;
    
    % (optional) clean small holes/specks
    obj.inc_mask_def  = imfill(obj.inc_mask_def,'holes');  obj.inc_mask_def  = bwareaopen(obj.inc_mask_def, 8);
    obj.coop_mask_def = imfill(obj.coop_mask_def,'holes'); obj.coop_mask_def = bwareaopen(obj.coop_mask_def, 8);


      % quick placement printout
      rows = find(any(obj.inc_mask, 2));  cols = find(any(obj.inc_mask, 1));
      z_mm = (rows - 1) * dx * 1e3;       % from top of grid
      y_mm = y_vec(cols) * 1e3;
      fprintf('Inclusion depth:  [%.1f .. %.1f] mm\n', min(z_mm), max(z_mm));
      fprintf('Inclusion lateral:[%.1f .. %.1f] mm\n', min(y_mm), max(y_mm));
    end

    function [med0, med1] = getMediumStructs(obj)
      % Baseline
      med0 = struct();
      med0.sound_speed = obj.sound_speed;
      med0.density     = obj.density;
      med0.alpha_power = obj.alpha_power;
      med0.alpha_coeff = obj.alpha_coeff;
      med0.alpha_mode  = 'no_dispersion';

      % Deformed
      med1 = struct();
      med1.sound_speed = obj.sound_speed_def;
      med1.density     = obj.density_def;
      med1.alpha_power = obj.alpha_power;
      med1.alpha_coeff = obj.alpha_coeff_def;
      med1.alpha_mode  = 'no_dispersion';
    end
  end
end
