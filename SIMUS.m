% clear;clc
% gpuDevice([]);
% use_single = true;
% % I = rgb2gray(imread('carotid.jpg'));
% I = load('P39-W2-S5-T.mat').TumorArea;
% I = imresize(I,[256 256]);
% 
% % 2) give background a small floor so it can speckle
% bg_floor = 0.03;            % try 0.02–0.06
% Gf = max(I, bg_floor);
% 
% % 3) multiplicative Rayleigh speckle everywhere
% speckle_strength = 0.8;     % 0=no speckle, 1=full
% sigma = sqrt(2/pi);         % unit-mean Rayleigh scale
% R = sigma * sqrt(-2*log(max(rand(size(Gf)), eps)));
% M = (1 - speckle_strength) + speckle_strength * R;
% S = Gf .* M;
% 
% I = mat2gray(log(S+1e-6));
% 
% % 4) quick displays
% subplot(1,2,1); imshow(mat2gray(S)); title('Linear');
% subplot(1,2,2); imshow(I); title('Log (B-mode-ish)');
% colormap gray
% 
% %%
% param = getparam('L11-5v');
% % param.fs = 2*(param.fc*(1+ (param.bandwidth/100)/2))*1.4;
% param.fs = 4*param.fc;
% param.c0 = 1540;
% 
% [xs,~,zs,RC] = genscat([3e-2 NaN],param,I,0.5);
% 
% dels = zeros(1,param.Nelements); % all delays are zero (plane wave)
% opt.WaitBar = false;
% tic
% % RF1 = simus(xs,zs,RC,dels,param,opt);
% % t1 = toc; % computation time for SIMUS
% 
% disp('REMOVING SCATTERERS')
% tic
% [xs2,zs2,RC2] = rmscat(xs,zs,RC,dels,param);
% t2 = toc; % computation time for RMSCAT + SIMUS
% disp('DONE REMOVING SCATTERERS')
% 
% %%
% subplot(211)
% scatter(xs*1e2,zs*1e2,2,20*log10(RC/max(RC(:))),'filled')
% ylabel('[cm]')
% clim([-40 0]) % range = [-40 0] dB
% axis ij tight equal
% title([int2str(numel(RC)) ' scatterers'])
% set(gca,'XColor','none','box','off')
% subplot(212)
% scatter(xs2*1e2,zs2*1e2,2,20*log10(RC2/max(RC2(:))),'filled')
% ylabel('[cm]')
% clim([-40 0]) % range = [-40 0] dB
% axis ij tight equal
% title({'after RMSCAT:',[int2str(numel(RC2)) ' scatterers']})
% set(gca,'XColor','none','box','off')
% colormap(hot)
% 
% %%
% % disp('RUNNING SIMUS...')
% % tic
% % [RF2,param] = simus(xs2,zs2,RC2,dels,param,opt);
% % t2 = t2 + toc;
% % disp('DONE SIMUS')
% % 
% % %%
% % IQ2 = rf2iq(RF2,param.fs,param.fc);
% % 
% % %%
% % [x,z] = meshgrid(linspace(-2e-2,2e-2,256),linspace(eps,3e-2,200));
% % param.fnumber = [];
% % IQ2b = das(IQ2,x,z,dels,param);
% % 
% % %%
% % I2 = bmode(IQ2b,50);
% % 
% % %%
% % % subplot(211)
% % % imagesc(x(1,:)*100,z(:,1)*100,I1)
% % % ylabel('[cm]')
% % % axis equal ij tight
% % % title(['computation time: ' int2str(t1) ' s'])
% % % subplot(212)
% % figure;
% % imagesc(x(1,:)*100,z(:,1)*100,I2)
% % ylabel('[cm]')
% % axis equal ij tight
% % title({'(with RMSCAT)',['computation time: ' int2str(t2) ' s']})
% % colormap gray
% 
% %% Focused linear array scanning (2500 x 256), minimal change
% disp('RUNNING SIMUS (focused, line-by-line)...')
% t2 = 0;                            % reset timing accumulator
% Nt = 2500;
% 
% % --- Array geometry (build element x-positions, centered at 0) ---
% Ne = param.Nelements;
% xe = ((0:Ne-1) - (Ne-1)/2) * param.pitch;   % [m], 1xNe
% 
% % --- Output grids: 256 laterals, 2500 axials ---
% xL = linspace(-2e-2, 2e-2, 256);            % lateral FOV [m] (match your old range)
% zax = linspace(eps, 3e-2, 2500).';          % axial grid [m] (2500 rows)
% 
% % --- Fixed TX focus depth (edit if desired) ---
% zf = 20e-3;                                 % 20 mm focus
% 
% % --- Preallocate envelope image (axial x lateral) ---
% ENV = zeros(numel(zax), numel(xL), 'single');
% 
% % --- RF time base (from param or infer from your SIMUS setup) ---
% t_rf = (0:Nt-1) / param.fs;           % seconds
% 
% % --- Options to SIMUS (keep your 'opt') ---
% opt.WaitBar = false;
% 
% tic
% for il = 1:numel(xL)
%     xl = xL(il);
% 
%     % 1) Per-line TX delays to focus at (xl, zf)
%     dtx   = sqrt((xe - xl).^2 + zf^2);      % geometric distance [m] per element
%     tauTX = (max(dtx) - dtx) / param.c0;    % delays [s]; outer fires earlier
% 
%     dels = tauTX;                            % SIMUS delays for this transmit
% 
%     % 2) Single focused transmit → channel RF
%     RF_line = simus(xs2, zs2, RC2, dels, param, opt);   % Nt x Ne
% 
%     % 3) Demod to I/Q channels (baseband) for this transmit
%     IQ_line = rf2iq(RF_line, param.fs, param.fc);       % Nt x Ne (complex)
% 
%     % 4) DAS to one lateral position (this line) over the axial grid
%     %    Use your existing 'das' but pass the single lateral position 'xl'
%     [x_one, z_col] = meshgrid(xl, zax);                 % (2500x1)
%     IQ_col = das(IQ_line, x_one, z_col, dels, param);   % 2500 x 1 (complex)
% 
%     % 5) Envelope (magnitude) for this column
%     ENV(:, il) = single(abs(IQ_col));
%     fprintf('%d / %d\n',il,numel(xL))
% end
% t2 = t2 + toc;
% disp('DONE SIMUS (focused)')
% 
% % --- Log compression & display ---
% I2 = bmode(ENV, 50);   % 50 dB dynamic range (same as you used)
% 
% figure;
% imagesc(xL*100, zax*1000, I2)           % lateral in cm, depth in mm (pick your favorite)
% xlabel('Lateral [cm]'); ylabel('Depth [mm]');
% axis equal ij tight
% title({'Focused linear array (with RMSCAT)', ['computation time: ' int2str(t2) ' s']})
% colormap gray


% %% --------- Focused linear array: fastest GPU DAS (2500 x 256) ---------
% disp('RUNNING SIMUS (focused, GPU DAS)...')
% t2 = 0;
% 
% % Sizes
% Nax  = 2500;                      % axial samples (rows)
% Nlat = 256;                       % lateral lines (cols)
% 
% % Array geometry (centered at 0)
% Ne = param.Nelements;
% xe = ((0:Ne-1) - (Ne-1)/2) * param.pitch;    % [m], 1xNe
% 
% % Imaging grids
% xL  = linspace(-2e-2,  2e-2,  Nlat);         % lateral FOV [m]
% zax = linspace(eps, 3e-2, Nax).';            % axial grid [m] (column)
% 
% % Transmit focus (single zone)
% zf = 20e-3;                                   % 20 mm focus
% 
% % F-number aperture (limits active elements; major speedup)
% Fnum = 1.7;                                    % 1.5–2.0 is typical
% 
% % Precompute GPU constants (single precision)
% xeg   = gpuArray(single(xe));                  % 1 x Ne
% zaxg  = gpuArray(single(zax));                 % Nax x 1
% apodg = gpuArray(single(hanning(Ne)));         % Ne x 1
% 
% % Output buffer on GPU (we'll gather once at the end)
% ENVg = gpuArray.zeros(Nax, Nlat, 'single');
% IQIMG = complex(zeros(numel(zax), numel(xL), 'single'));
% 
% tic
% wb = waitbar(0,'Running Simulation');
% for il = 1:Nlat
%     waitbar(il/Nlat, wb, sprintf('Running Simulation: %d / %d', il, Nlat));
%     xl = xL(il);
% 
%     % --- TX delays for focus at (xl, zf) on CPU (tiny math) ---
%     dtx   = sqrt((xe - xl).^2 + zf^2);                 % [m]
%     tauTX = (max(dtx) - dtx) / param.c0;               % [s]
%     dels  = tauTX;                                     % SIMUS delays
% 
%     % --- SIMUS (CPU). Returns Nt x Ne channel RF (double) ---
%     RF_line = simus(xs2, zs2, RC2, dels, param, opt);
% 
%     % --- Demod to I/Q and move to GPU as single ---
%     IQ_line = single(rf2iq(RF_line, param.fs, param.fc));  % Nt x Ne (complex single)
%     IQg     = gpuArray(IQ_line);
% 
%     % --- GPU DAS: vectorized interpolation + sum ---
%     Nt  = size(IQg,1);
%     fs  = single(param.fs);
%     c0  = single(param.c0);
% 
%     % For every depth & element: time-of-flight including TX delay
%     % dz: Nax x Ne (via implicit expansion), xl on GPU
%     xlg   = gpuArray(single(xl));
%     dz    = sqrt( (xeg - xlg).^2 + zaxg.^2 );            % [m], Nax x Ne
%     tSamp = single(tauTX) + 2*dz/c0;                     % [s],  Nax x Ne
% 
%     % Convert times to fractional sample index s in [1..Nt]
%     s   = tSamp*fs + 1;                                  % Nax x Ne
%     i0  = max(1, min(Nt-1, floor(s)));                   % lower index
%     a   = s - i0;                                        % fractional part
% 
%     % Build linear indices (no loops)
%     ne   = int32(size(IQg,2));
%     Nt_i = int32(Nt);
%     i0i  = int32(i0);
%     off  = gpuArray(int32(0:ne-1) * Nt_i);                % 1 x Ne
%     idx0 = i0i + off;                                    % Nax x Ne
%     idx1 = idx0 + 1;
% 
%     % Gather neighbor samples and linearly interpolate
%     RF0 = IQg(idx0);                                     % Nax x Ne
%     RF1 = IQg(idx1);
%     V   = (1 - a).*RF0 + a.*RF1;                         % Nax x Ne
% 
%     % F-number aperture mask per depth (Nax x Ne)
%     aper_half = zaxg / single(Fnum);                     % [m]
%     mask = abs(xeg - xlg) <= aper_half;                  % logical Nax x Ne
% 
%     % Apply RX apod + mask, then sum elements
%     V = V .* (apodg.' .* mask);                          % broadcast Ne over rows
%     col = sum(V, 2);                                     % Nax x 1 complex
% 
%     % Envelope (magnitude) → store
%     ENVg(:, il) = abs(col);
% end
% t2 = t2 + toc;
% disp('DONE SIMUS (focused, GPU DAS)')
% 
% % Log compression & display (gather once)
% ENV = gather(ENVg);
% I2  = bmode(ENV, 50);
% 
% figure;
% imagesc(xL*100, zax*1000, I2)
% xlabel('Lateral [cm]'); ylabel('Depth [mm]');
% axis equal ij tight
% title({'Focused linear array (GPU DAS + F/#)', ['time: ' num2str(t2,'%.2f') ' s']})
% colormap gray
% %%
% %% ===== Quick QA on current run (NO RERUN NEEDED) =====
% % Needs: ENV (axial x lateral, real), xL (1xN), zax (Nx1), I (256x256 or similar)
% 
% % 0) Guard rails
% assert(exist('ENV','var')==1, 'ENV not found in workspace.');
% assert(isreal(ENV), 'ENV must be real (you stored |IQ|).');
% [na, nl] = size(ENV);
% fprintf('ENV size: %d (axial) x %d (lateral)\n', na, nl);
% 
% % 1) Basic sanity
% n_nan = sum(~isfinite(ENV(:)));
% fprintf('NaN/Inf count: %d\n', n_nan);
% mx = max(ENV(:)); mn = min(ENV(:));
% fprintf('Amplitude range: [%.3g, %.3g]\n', mn, mx);
% ENVn = ENV / max(mx, eps);  % normalize to [0,1]
% 
% % 2) Build a rough inclusion mask from your texture I (just for QA)
% if ~exist('I','var'); error('Texture I not found'); end
% Itex = im2double(I);
% Itex = imresize(Itex, [na nl], 'nearest');     % to 2500x256
% th   = graythresh(Itex);                       % Otsu
% Mask = Itex >= th;
% % clean mask a bit
% Mask = bwareaopen(Mask, 50);
% Mask = imfill(Mask, 'holes');
% 
% % 3) ROI stats in linear envelope
% in_vals = ENVn(Mask);
% bg_vals = ENVn(~Mask & isfinite(ENVn));
% 
% mu_in = mean(in_vals);  sd_in = std(in_vals);
% mu_bg = mean(bg_vals);  sd_bg = std(bg_vals);
% CNR   = abs(mu_in - mu_bg) / sqrt(sd_in^2 + sd_bg^2);
% ENL_bg = (mu_bg / max(sd_bg, eps))^2;   % Rayleigh-ish ENL
% 
% fprintf('Inclusion mean=%.4f, std=%.4f\n', mu_in, sd_in);
% fprintf('Background mean=%.4f, std=%.4f\n', mu_bg, sd_bg);
% fprintf('CNR=%.3f, ENL_bg≈%.2f (Rayleigh ideal ~1)\n', CNR, ENL_bg);
% 
% % 4) Axial and lateral profiles
% col_mid = round(nl/2);
% aline   = ENVn(:, col_mid);
% lat_prof= mean(ENVn, 1);
% 
% % 5) Depth support vs your grid (checks you didn't truncate or pad)
% if exist('zax','var') && numel(zax)==na
%     z_span_mm = [zax(1) zax(end)]*1e3;
%     fprintf('Depth axis spans %.1f–%.1f mm\n', z_span_mm(1), z_span_mm(2));
% else
%     warning('zax missing/mismatched; skipping depth print.');
% end
% if exist('xL','var') && numel(xL)==nl
%     x_span_mm = [xL(1) xL(end)]*1e3;
%     fprintf('Lateral axis spans %.1f–%.1f mm\n', x_span_mm(1), x_span_mm(2));
% end
% 
% % 6) Quick displays (no log, since ENV is already magnitude)
% figure('Name','QA (no rerun)');
% tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
% 
% nexttile;
% imagesc(xL*1e3, zax*1e3, ENVn); axis ij image tight; colormap gray
% xlabel('Lateral (mm)'); ylabel('Depth (mm)');
% title('Linear envelope (normalized)');
% 
% nexttile;
% imagesc(xL*1e3, zax*1e3, ENVn); axis ij image tight; colormap gray
% hold on; contour(xL*1e3, zax*1e3, Mask, [0.5 0.5], 'r', 'LineWidth', 0.8);
% xlabel('Lateral (mm)'); ylabel('Depth (mm)');
% title('Envelope + inclusion mask (rough)');
% 
% nexttile;
% plot(zax*1e3, aline); grid on
% xlabel('Depth (mm)'); ylabel('|A| (norm)');
% title(sprintf('Mid A-line (col %d)', col_mid));
% 
% nexttile;
% plot(xL*1e3, lat_prof); grid on
% xlabel('Lateral (mm)'); ylabel('Mean |A|');
% title('Mean lateral profile');
% 
% drawnow;
% 
% % 7) Optional: background Rayleigh check (quick-and-dirty)
% nbg = min(2e5, numel(bg_vals));
% h = histogram(bg_vals(randperm(numel(bg_vals), nbg)), 80, 'Normalization','pdf'); hold on
% x = linspace(0, max(h.BinEdges), 400);
% sigma_hat = sqrt(mean(bg_vals.^2)/2);
% pdf_ray = (x./sigma_hat.^2).*exp(-x.^2/(2*sigma_hat^2));
% plot(x, pdf_ray, 'r', 'LineWidth', 1.25); hold off
% title('BG amplitude PDF vs Rayleigh fit'); xlabel('|A|'); ylabel('pdf');


%% --------- Focused linear array: fastest GPU DAS (2500 x 256) ---------
disp('RUNNING SIMUS (focused, GPU DAS)...')
t2 = 0;

% Sizes
Nax  = 2500;                                % axial samples (rows)
Nlat = 96;                                 % lateral lines (cols)

% Array geometry (centered at 0)
Ne = param.Nelements;
xe = ((0:Ne-1) - (Ne-1)/2) * param.pitch;   % [m], 1xNe

% Imaging grids
xL  = linspace(-2e-2,  2e-2,  Nlat);        % lateral FOV [m]
zax = linspace(eps, 3e-2, Nax).';           % axial grid [m] (column)

% Transmit focus (single zone)
zf = 20e-3;                                  % 20 mm focus

% F-number aperture (limits active elements; major speedup)
Fnum = 1.7;                                  % 1.5–2.0 is typical

% --- k-Wave/Field-II style scatterer pruning knob ---
% keep scatterers within ±optimization_coeff * total lateral span around the line
lat_span = max(xs2) - min(xs2);                                    % total scatterer lateral span
K = 3;                                                    % roughly ±K * pitch
optimization_coeff = max( (K*param.pitch)/max(lat_span,eps), 0.06 );  % ~0.06–0.15 works well
zmin = zax(1); zmax = zax(end);

% Precompute GPU constants (single precision)
xeg   = gpuArray(single(xe));               % 1 x Ne
zaxg  = gpuArray(single(zax));              % Nax x 1
apodg = gpuArray(single(hanning(Ne)));      % Ne x 1

% Output buffers on GPU (gather once at the end)
IQgImg = gpuArray(complex(zeros(Nax, Nlat, 'single')));   % COMPLEX IQ (for bmode)
ENVg   = gpuArray(zeros(Nax, Nlat, 'single'));            % envelope (optional debug)

tic
for il = 1:Nlat
    xl = xL(il);

    if mod(il,2) == 0
        fprintf('Progress: %d / %d\n',il,Nlat);
    end

    % --- TX delays for focus at (xl, zf) on CPU (tiny math) ---
    dtx   = sqrt((xe - xl).^2 + zf^2);                   % [m]
    tauTX = (max(dtx) - dtx) / param.c0;                 % [s]
    dels  = tauTX;                                       % SIMUS delays

    % --- Field-II/k-Wave style lateral pruning (per line) ---
    start_range = xl - optimization_coeff * lat_span;
    end_range   = xl + optimization_coeff * lat_span;
    keep = (xs2 > start_range) & (xs2 < end_range) & (zs2 >= zmin) & (zs2 <= zmax);
    if ~any(keep), keep = true(size(xs2)); end           % safety
    xsL = xs2(keep);  zsL = zs2(keep);  RCL = RC2(keep);

    % --- SIMUS (CPU). Returns Nt x Ne channel RF (double) ---
    RF_line = simus(xsL, zsL, RCL, dels, param, opt);

    % --- Demod to I/Q and move to GPU as single ---
    IQ_line = single(rf2iq(RF_line, param.fs, param.fc));    % Nt x Ne (complex single)
    IQg     = gpuArray(IQ_line);

    % --- GPU DAS: vectorized interpolation + sum ---
    Nt  = size(IQg,1);
    fs  = single(param.fs);
    c0  = single(param.c0);

    % For every depth & element: time-of-flight including TX delay
    xlg   = gpuArray(single(xl));
    dz    = sqrt( (xeg - xlg).^2 + zaxg.^2 );                % [m], Nax x Ne (implicit expansion)
    tSamp = single(tauTX) + 2*dz/c0;                         % [s],  Nax x Ne

    % Convert times to fractional sample index s in [1..Nt]
    s  = tSamp*fs + 1;                                       % Nax x Ne
    i0 = max(1, min(Nt-1, floor(s)));                        % lower index
    a  = s - i0;                                             % fractional part

    % ---- build linear indices (use doubles on GPU; fastest & simple) ----
    ne   = size(IQg,2);
    i0d  = gpuArray(double(i0));                             % Nax x Ne
    off  = gpuArray(double(0:ne-1)) * double(Nt);            % 1 x Ne
    idx0 = i0d + off;                                        % Nax x Ne
    idx1 = idx0 + 1;

    % Gather neighbor samples and linearly interpolate
    RF0 = IQg(idx0);                                         % Nax x Ne
    RF1 = IQg(idx1);
    V   = (1 - a).*RF0 + a.*RF1;                             % Nax x Ne

    % F-number aperture mask per depth (Nax x Ne)
    aper_half = zaxg / single(Fnum);                         % [m]
    mask = abs(xeg - xlg) <= aper_half;                      % logical Nax x Ne

    % Apply RX apod + mask, then sum elements
    V   = V .* (apodg.' .* mask);                            % broadcast Ne over rows
    col = sum(V, 2);                                         % Nax x 1 complex

    % --- store BOTH: complex IQ column and envelope column ---
    IQgImg(:, il) = col;                                     % complex (for bmode)
    ENVg(:,  il)  = abs(col);                                % real    (debug)
end
t2 = t2 + toc;
if isvalid(wb), delete(wb); end
disp('DONE SIMUS (focused, GPU DAS)')

% ---- Gather once, make B-mode properly from COMPLEX IQ ----
IQIMG = gather(IQgImg);                      % complex [2500 x 256]
ENV   = gather(ENVg);                        % real envelope (debug)

IQIMG256 = imresize(IQIMG, [2500 256]);  % complex-safe
I2       = bmode(IQIMG, 50);
I2 = imresize(I2,[256 256]);

figure;
imagesc(I2)
xlabel('Lateral [cm]'); ylabel('Depth [mm]');
axis equal ij tight
title({'Focused linear array (GPU DAS + F/#)', ['time: ' num2str(t2,'%.2f') ' s']})
colormap gray


%% ===== Quick QA on current run =====
% Uses IQIMG (complex), ENV (real), xL, zax, I (your 256x256 texture)

% 0) Guard rails
assert(exist('IQIMG','var')==1 && ~isreal(IQIMG), 'IQIMG (complex) missing.');
[na, nl] = size(IQIMG);
fprintf('IQIMG size: %d (axial) x %d (lateral)\n', na, nl);

% 1) Basic sanity on envelope
ENVn = ENV / max(ENV(:)+eps);      % normalize to [0,1]
n_nan = sum(~isfinite(ENVn(:)));
fprintf('NaN/Inf count in ENVn: %d\n', n_nan);

% 2) Rough inclusion mask from your texture
Itex = im2double(I);
Itex = imresize(Itex, [na nl], 'nearest');
th   = graythresh(Itex);
Mask = imfill(bwareaopen(Itex >= th, 50), 'holes');

% 3) ROI stats in linear envelope
in_vals = ENVn(Mask);
bg_vals = ENVn(~Mask & isfinite(ENVn));
mu_in = mean(in_vals);  sd_in = std(in_vals);
mu_bg = mean(bg_vals);  sd_bg = std(bg_vals);
CNR   = abs(mu_in - mu_bg) / sqrt(sd_in^2 + sd_bg^2);
ENL_bg= (mu_bg / max(sd_bg, eps))^2;

fprintf('Inclusion mean=%.4f, std=%.4f | Background mean=%.4f, std=%.4f\n', mu_in, sd_in, mu_bg, sd_bg);
fprintf('CNR=%.3f, ENL_bg≈%.2f (Rayleigh ideal ~1)\n', CNR, ENL_bg);

% 4) Quick plots
figure('Name','QA after focused GPU DAS');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(xL*1e3, zax*1e3, I2); axis ij image tight; colormap gray
xlabel('Lateral (mm)'); ylabel('Depth (mm)'); title('B-mode (50 dB)'); colorbar;

nexttile;
imagesc(xL*1e3, zax*1e3, ENVn); axis ij image tight; colormap gray
hold on; contour(xL*1e3, zax*1e3, Mask, [0.5 0.5], 'r', 'LineWidth', 0.8);
xlabel('Lateral (mm)'); ylabel('Depth (mm)'); title('Envelope + inclusion mask');

nexttile;
col_mid = round(nl/2); plot(zax*1e3, ENVn(:,col_mid)); grid on
xlabel('Depth (mm)'); ylabel('|A| (norm)'); title(sprintf('Mid A-line (col %d)', col_mid));

nexttile;
plot(xL*1e3, mean(ENVn,1)); grid on
xlabel('Lateral (mm)'); ylabel('Mean |A|'); title('Mean lateral profile');
drawnow;
