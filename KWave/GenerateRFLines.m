function [scan_lines, rf, rf_raw] = GenerateRFLines(Nx, Ny, dy, num_scan_lines, sound_speed_map, density_map, y_centers_local_m, focus_depth, x_src_m, ...
    c0, kgrid, DATA_CAST, num_elems, tx_apo, burst, y_centers_pix, muteN, elem_width_gdpts, pml_x_size, pml_y_size, steer_deg, burstN, x_src_pix)

scan_lines = zeros(num_scan_lines, kgrid.Nt, DATA_CAST);

input_args = {'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], ...
              'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

medium_position = 1;

% Medium (background)
medium.alpha_coeff = 0.75;            % dB/(MHz^y·cm)
medium.alpha_power = 1.5;
medium.BonA = 6;                      % enable nonlinearity (THI path uses it)

for li = 1:num_scan_lines
        fprintf('Computing scan line %d / %d\n', li, num_scan_lines);

        % slice this Ny window from the global phantom
        % TOP IS THE SPEED IMPROVEMENT VERSION
        medium.sound_speed = sound_speed_map(:, medium_position:medium_position+Ny-1);
        medium.density     = density_map(:,   medium_position:medium_position+Ny-1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%5
        % medium.sound_speed = sound_speed_map;
        % medium.density = density_map;  % Set the density for the medium
        %%%%%%%%%%%%%%%%%%%%%%%%%%%

        disp('Sound speed size')
        disp(size(medium.sound_speed));  % Display the size of the sound speed matrix
        disp('kgrid sizes')
        disp(kgrid.Nx);  % Display the size of the kgrid structure
        disp(kgrid.Ny)

        % assert(isequal(size(sound_speed_map), [kgrid.Nx, kgrid.Ny]));
        % assert(isequal(size(density_map),     [kgrid.Nx, kgrid.Ny]));



        % -------- Global lateral coordinates for this window ----------
        % TOP 2 ARE THE CHANGE FOR SPEED IMPROVEMENT
        y0_m = (medium_position - 1) * dy;                       % global origin of window
        y_centers_global_m = y0_m + y_centers_local_m;           % element centres in global metres
        % y_centers_global_m = (y_centers_pix-1) * dy;  % no global offset

        % beam lateral (global) = mid of window (+ steering offset)
        % TOP 2 ARE SPEED IMPROVEMENT
        y_focus_pix = medium_position - 1 + round((Ny+1)/2);
        y_focus_m   = (y_focus_pix - 1)*dy + focus_depth * tand(steer_deg);
        % y_focus_m = ((Ny+1)/2 - 1)*dy + focus_depth * tand(steer_deg);


        % -------- Build per-element TX drive (global focus delays) ----
        r_e = sqrt( (focus_depth - x_src_m).^2 + (y_centers_global_m - y_focus_m).^2 );
        tau = (max(r_e) - r_e) / c0;          % delay to make all waves meet at same time
        drive = zeros(num_elems, kgrid.Nt, DATA_CAST);
        for e = 1:num_elems
            idx0 = 1 + round(tau(e) / kgrid.dt);
            if idx0 <= kgrid.Nt
                nAvail = min(burstN, kgrid.Nt - idx0 + 1);
                drive(e, idx0:idx0+nAvail-1) = tx_apo(e) * burst(1:nAvail);
            end
        end

        % particle velocity mask at element centres (one cell per element)
        % TOP 2 ARE FOR SPEED
        u_mask = false(Nx, Ny);
        u_mask(x_src_pix, y_centers_pix) = true;   

        source = struct();
        source.u_mask = u_mask;
        source.ux     = drive;        % rows: elements (mask order), cols: time

        % receive the same points
        sensor = struct();
        sensor.mask   = u_mask;
        sensor.record = {'p'};

        % propagate
        raw = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        clc
        rf_raw  = raw.p;                                  % size = [num_rx x Nt]
        num_rx = size(rf_raw,1);

        % channel-wise DC / early-time mute
        rf_raw(:, 1:muteN) = 0;
        rf = rf_raw - mean(rf_raw,2);

        % -------- Delay-and-sum (global delays) -----------------------
        rx_apo = ones(1, num_rx);                     % rectangular receive apod
        scan   = zeros(1, kgrid.Nt, DATA_CAST);
        rows   = (1:num_rx).';                        % for sub2ind
        z_m    = (0:kgrid.Nt-1) * kgrid.dt * c0/2;    % depth (two-way)

        % choose TX phase centre (full aperture mean)
        tx_yc = mean(y_centers_global_m);
        tx_xc = x_src_m;

        for k = 1:kgrid.Nt
            y_line_m = y_focus_m;                     % lateral location of this scan line
            r_tx = sqrt( (z_m(k) - tx_xc).^2 + (y_line_m - tx_yc).^2 );      % TX one-way
            r_rx = sqrt( (z_m(k) - x_src_m).^2 + (y_centers_global_m - y_line_m).^2 ); % RX
            tof  = (r_tx + r_rx) / c0;                                        % seconds

            idxf = 1 + tof / kgrid.dt;                % fractional sample indices
            i0   = max(1, min(floor(idxf), kgrid.Nt-1));
            a    = idxf - i0;                         % linear interp weight
            i1   = i0 + 1;

            v0 = rf(sub2ind([num_rx, kgrid.Nt], rows, i0(:)));
            v1 = rf(sub2ind([num_rx, kgrid.Nt], rows, i1(:)));
            vals = (1 - a(:)).*v0 + a(:).*v1;

            scan(k) = sum(rx_apo(:) .* vals);
        end

        % apply mute once to the finished A-line
        scan(1:muteN) = 0;

        % store
        scan_lines(li,:) = gather_if_needed(scan, DATA_CAST);

        % move the window one element-width to the right
        medium_position = medium_position + elem_width_gdpts;
end

% ------------ helper ------------
function out = gather_if_needed(x, CAST)
    if contains(CAST,'gpu'), out = gather(x); else, out = x; end
end
end