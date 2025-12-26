function metrics = CompareWindowToRegularRFTechnique(rf_A, rf_B, show_figs)
% Compare two RF datasets (num_scanlines x Nt) from k-Wave pipelines.
% - Robust envelope detection (along time, dim 2)
% - Per-line normalized cross-correlation allowing small lags
% - NMSE on RF, SSIM/PSNR on log-compressed envelopes
% - Per-line lag and correlation distributions (optional plots)
%
% Usage:
%   metrics = CompareRFVariants(scan_lines_fast, scan_lines_ref, true);

    if nargin < 3, show_figs = false; end
    t0 = tic;

    % ---------- Basic checks ----------
    assert(isequal(size(rf_A), size(rf_B)), 'Size mismatch between RF datasets.');
    [num_lines, Nt] = size(rf_A);

    % Ensure double for stable numerics
    rf_A = double(rf_A);
    rf_B = double(rf_B);

    % ---------- Envelope detection along time (dim 2) ----------
    % hilbert operates along columns; transpose, apply, transpose back
    env_A = abs(hilbert(rf_A.')).';
    env_B = abs(hilbert(rf_B.')).';

    % ---------- Log-compressed B-mode images (shared reference) ----------
    eps_val = 1e-12;
    ref_max = max(env_B(:)) + eps_val;        % normalize BOTH to B (reference) max
    imgA = 20*log10(env_A/ref_max + eps_val);
    imgB = 20*log10(env_B/ref_max + eps_val);

    % ---------- RF-domain errors ----------
    nmse_rf = norm(rf_A(:) - rf_B(:))^2 / (norm(rf_B(:))^2 + eps_val);

    % Optional: amplitude-compensated NMSE per line (helps if global gain differs)
    nmse_rf_gain = zeros(num_lines,1);
    for i = 1:num_lines
        a = rf_A(i,:); b = rf_B(i,:);
        % optimal scalar scale for a ≈ s*b (least squares)
        s = (a*b.') / (b*b.' + eps_val);
        nmse_rf_gain(i) = norm(a - s*b)^2 / (norm(b)^2 + eps_val);
    end
    nmse_rf_gain_mean = mean(nmse_rf_gain);

    % ---------- Per-line NCC with small lag search ----------
    % Allow up to ~3–5 samples of drift; make it a fraction of Nt but capped
    maxlag = min(32, floor(Nt/10));
    ncc_max   = zeros(num_lines,1);
    ncc_lag   = zeros(num_lines,1);
    ncc_0lag  = zeros(num_lines,1);
    for i = 1:num_lines
        a = rf_A(i,:); b = rf_B(i,:);
        [c,lags] = xcorr(a, b, maxlag, 'normalized');
        [ncc_max(i), idx] = max(c);
        ncc_lag(i) = lags(idx);
        % also record zero-lag correlation (can be negative if phase differs)
        ncc_0lag(i) = corr(a(:), b(:), 'Rows','complete');
    end
    ncc_mean         = mean(ncc_max);
    ncc0_mean        = mean(ncc_0lag);
    mean_abs_lag_smp = mean(abs(ncc_lag));

    % ---------- Image-domain similarity ----------
    % SSIM (Image Processing Toolbox). Fallback if missing.
    try
        ssim_val = ssim(mat2gray(imgA), mat2gray(imgB));
    catch
        % Fallback: NCC of log envelopes as a crude proxy
        tmpA = imgA(:); tmpB = imgB(:);
        ssim_val = corr(tmpA, tmpB, 'Rows','complete');
    end

    % PSNR on log images (manual, no toolbox dependency)
    mse_img = mean( (imgA(:) - imgB(:)).^2 );
    % dynamic range ~ 60 dB is common; here use data-driven peak
    peak   = max(max(imgA(:)), max(imgB(:)));
    psnr_db = 10*log10( (peak^2) / (mse_img + eps_val) );

    % ---------- Pack results ----------
    metrics = struct();
    metrics.nmse_rf              = nmse_rf;
    metrics.nmse_rf_gain_mean    = nmse_rf_gain_mean;
    metrics.ncc_mean_maxlag      = ncc_mean;          % mean of per-line max NCC
    metrics.ncc_mean_zerolag     = ncc0_mean;         % mean of per-line zero-lag corr
    metrics.per_line_ncc_max     = mean(ncc_max);
    metrics.per_line_ncc_lag     = mean(ncc_lag);           % samples (positive = A lags B)
    metrics.mean_abs_lag_samples = mean_abs_lag_smp;
    metrics.ssim                 = ssim_val;
    metrics.psnr_db              = psnr_db;
    metrics.elapsed_s            = toc(t0);

    % ---------- Optional visualization ----------
    if show_figs
        % B-modes
        figure('Name','B-mode (log env, dB)'); 
        subplot(1,2,1); imagesc(imgB); axis image; colormap gray; title('Reference (B)'); colorbar;
        subplot(1,2,2); imagesc(imgA); axis image; colormap gray; title('Variant (A)'); colorbar;

        % NCC and lag
        figure('Name','Per-line NCC / Lag');
        subplot(3,1,1); plot(ncc_max,'LineWidth',1); ylim([0,1]); grid on;
        ylabel('max NCC'); title(sprintf('Mean max NCC = %.3f', ncc_mean));
        subplot(3,1,2); stem(ncc_lag,'filled'); grid on;
        ylabel('lag (samples)'); title(sprintf('Mean |lag| = %.2f smp', mean_abs_lag_smp));
        subplot(3,1,3); plot(ncc_0lag,'LineWidth',1); ylim([-1,1]); grid on;
        ylabel('zero-lag corr'); xlabel('Scanline');

        % Difference image (log env)
        figure('Name','Log-Envelope Difference (A - B)');
        imagesc(imgA - imgB); axis image; colormap gray; colorbar;
        title(sprintf('Δ log-env, PSNR=%.2f dB, SSIM=%.3f', psnr_db, ssim_val));
    end
end
