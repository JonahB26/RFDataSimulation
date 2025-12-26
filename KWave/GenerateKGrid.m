function [Nx, Ny, dx, dy, kgrid] = GenerateKGrid(Nx_eff, Ny_eff, x_span, pml_x_size, pml_y_size, c0)
Nx = Nx_eff - 2*pml_x_size;                % k-Wave sim window
Ny = Ny_eff - 2*pml_y_size;
%%%%%%%%%%%%%%%%%%%%%%%%%
% Ny = 364;
%%%%%%%%%%%%%%%%%%%%%%%%%

dx = x_span / Nx;  dy = dx;

kgrid = makeGrid(Nx, dx, Ny, dy);

% Time array (enough for round-trip)
t_end = (Nx*dx) * 2.2 / c0;
kgrid.t_array = makeTime(kgrid, c0, [], t_end);
end