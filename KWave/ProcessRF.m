function [scan_lines, scan_lines_fund, r] = ProcessRF(scan_lines, num_scan_lines, tgc_alpha, burstN, kgrid, tone_burst_freq, compression_ratio, scale_factor, c0)
% Early-time window (Tukey)
win = getWin(kgrid.Nt*2, 'Tukey', 'Param', 0.05).';
win = [zeros(1, burstN*2), win(1:end/2 - burstN*2)];
scan_lines = bsxfun(@times, win, scan_lines);

% TGC
t0 = burstN*kgrid.dt/2;
r  = c0*((1:length(kgrid.t_array))*kgrid.dt/2 - t0);   % range axis (m)
tgc = exp(2*tgc_alpha * tone_burst_freq/1e6 * r * 100);
scan_lines = bsxfun(@times, tgc, scan_lines);

% Fundamental and 2nd harmonic bandpass (Gaussian)
scan_lines_fund = gaussianFilter(scan_lines, 1/kgrid.dt, 1.5e6, 100, true);
% scan_lines_harm = gaussianFilter(scan_lines, 1/kgrid.dt, 3.0e6,  30, true);

% Envelope detection
scan_lines_fund = envelopeDetection(scan_lines_fund);
% scan_lines_harm = envelopeDetection(scan_lines_harm);

% Log compression
scan_lines_fund = logCompression(scan_lines_fund, compression_ratio, true);
% scan_lines_harm = logCompression(scan_lines_harm, compression_ratio, true);

% Lateral upsample for display
scan_lines_fund = interp2(1:kgrid.Nt, (1:num_scan_lines).', scan_lines_fund, ...
                          1:kgrid.Nt, (1:1/scale_factor:num_scan_lines).');
% scan_lines_harm = interp2(1:kgrid.Nt, (1:num_scan_lines).', scan_lines_harm, ...
%                           1:kgrid.Nt, (1:1/scale_factor:num_scan_lines).');
end