function out_mm = ConvertPXToMM(input_px)
% This function converts the input value from pixels into mm.
% Created by Jonah Boutin on 03/05/2025
dpi = get(0,"ScreenPixelsPerInch"); %Calculate dpi
pixels_per_mm = dpi / 25.4; %Convert to pixels/mm

out_mm = input_px ./ pixels_per_mm;   