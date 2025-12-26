function [burst, burstN, guardN, muteN] = GenerateInputSignal(kgrid, c0, rho0, source_strength, tone_burst_freq, tone_burst_cycles, guard_m)
burst = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);
burst = (source_strength / (c0 * rho0)) * burst;   % convert P -> particle velocity
burstN = numel(burst);

guardN  = round((2*guard_m/c0) / kgrid.dt);
muteN   = min(kgrid.Nt, burstN*2 + guardN);

end