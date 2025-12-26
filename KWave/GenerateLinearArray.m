function [x_src_pix, x_src_m, y_centers_pix, y_centers_local_m, tx_apo] = GenerateLinearArray(pml_x_size, dx, dy, Ny,num_elems, elem_width_gdpts, kerf_gdpts)
% array depth just below top PML
x_src_pix = pml_x_size + 6;
x_src_m   = (x_src_pix-1)*dx;

% lateral element centers (in pixels, local to the Ny window)
ap_width = num_elems*elem_width_gdpts + (num_elems-1)*kerf_gdpts;
y0_pix   = floor((Ny - ap_width)/2) + 1;                         % first column
y_centers_pix = y0_pix + (elem_width_gdpts+kerf_gdpts)*(0:num_elems-1) ...
                      + floor((elem_width_gdpts-1)/2);
y_centers_local_m = (y_centers_pix-1) * dy;                      % (local metres)

tx_apo = hann(num_elems).';
end