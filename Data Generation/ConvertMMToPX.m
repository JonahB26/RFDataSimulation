function out_px = ConvertMMToPX(input_mm)
% This function converts the input value from mm into pixels.
% Created by Jonah Boutin on 03/05/2025
dpi = get(0,"ScreenPixelsPerInch"); %Calculate dpi
pixels_per_mm = dpi / 25.4; %Convert to pixels/mm

out_px = pixels_per_mm * input_mm;   
