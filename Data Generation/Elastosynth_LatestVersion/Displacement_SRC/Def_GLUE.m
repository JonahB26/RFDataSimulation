function [Axial,Lateral,strainA,strainL] = Def_GLUE(Im1,Im2,coeffs,maxAL)
% Create displacements and strains using GLUE
%  Author: Hoda Sadat Hashemi - Modified by: Niusha Kheirkhah

IRF = [coeffs{1} coeffs{2}]; % Maximum allowed disparity in axial D
IA = [coeffs{3} coeffs{4}];  % Maximum allowed disparity in lateral D

xx = calc_att (coeffs{9}, coeffs{12}, coeffs{11}); % To compensate for attenuation
alfa_DP = 0.2; % DP regularization weight
tic
[ax0, lat0, ~] = RCF_AM2D(Im1, Im2, IRF, IA, coeffs{5}, alfa_DP, coeffs{6}, coeffs{7}, coeffs{8}, coeffs{10}, xx);
toc
% the disp. of the first 40 and last 40 samples is not calculated in AM2D: 
% the disp. of the first 15 and last 15 A-lines is not calculated in AM2D:
Axial = ax0(41:end-40, 16:end-15);
Lateral = lat0(41:end-40, 16:end-15);
Im1 = Im1(41:end-40, 16:end-15);
Im2 = Im2(41:end-40, 16:end-15);
alfa1 = coeffs{13} ; % Regularization coefficients
alfa2 = 1 ;
beta1 = coeffs{13} ; 
beta2 = 1 ;
%tic
[Axial, Lateral] = GLUE (Axial, Lateral, Im1, Im2, alfa1, alfa2, beta1, beta2);
%toc
% ---------------------------------------------------- %
% ------------ Calculating Strain from Disp ---------- %
% ---------------------------------------------------- %

maxA = 0.01 * floor(1-maxAL{1}) + maxAL{1};
maxL = 0.002 * floor(1-maxAL{2}) + maxAL{2};

% axial strain
wDIff = 3; % window length of the differentiation kernel
strainA = LSQ(Axial,wDIff);
strainA = -strainA((wDIff+1)/2:end-(wDIff-1)/2,(wDIff+1)/2:end-(wDIff-1)/2);
strainA(strainA == 0) = 0.0001;
strainA(strainA>maxA)=maxA;
% lateral strain:
strainL = LSQ(Lateral',wDIff);
strainL = strainL((wDIff+1)/2:end-(wDIff-1)/2,(wDIff+1)/2:end-(wDIff-1)/2)';
strainL(strainL == 0) = 0.0001;
strainL(strainL>maxL)=maxL;
strainA = abs(strainA);
strainL = abs(strainL);
end

