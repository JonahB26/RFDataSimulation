%% Define the input signal
% define properties of the input signal
source_strength = 1e6;          % [MPa]
tone_burst_freq = 0.5e6;        % [Hz]
tone_burst_cycles = 5;

% create the input signal using toneBurst
input_signal = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);