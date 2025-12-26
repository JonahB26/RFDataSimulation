%% GLUE
function GLUE = RunGLUE(Im1,Im2, params)
    
    endA = size(Im1,2);
    endRF = size(Im1,1);
    startA = 1;
    startRF = 1;
    L = params.L;
    D = params.D;

    xRat = L / (endA-1);
    yRat = D / (endRF-1);
    endA = endA - 30;
    endRF = endRF - 80;
    L = endA * xRat;
    D = endRF * yRat;
    dims = {endA,endRF,L,D,xRat,yRat};
    
    % Coefficients for the GLUE algorithm
    IRF1 = -35;  %1  DON't CHANGE % Maximum allowed disparity in axial D
    IRF2 = 35;    %2  DON't CHANGE
    IA1 = -4;    %3  DON't CHANGE % Maximum allowed disparity in lateral D
    IA2 = 4;     %4  DON't CHANGE
    midA = 50;  %5  MAY CHANGE
    alfa = 5;    %6  DON't CHANGE % Axial regularization
    beta = 5;   %7  DON't CHANGE % Lateral regularization
    gamma = 0.005;%8  DON't CHANGE % Lateral regularization
    T = .2; % threshold for IRLS
    a_t = params.probe.a_t; % frequency dependent attenuation coefficient, in dB/cm/MHz
    f_0 = params.probe.fc; %ultrasound center freq. in MHz
    f_s = params.probe.fs; % sampling freq. in MHz
    smth = 5;    %13 DON't CHANGE % Degree of smoothness
    coeffs = {IRF1,IRF2,IA1,IA2,midA,alfa,beta,gamma,a_t,T,f_s,f_0,smth};
    
    [Axial,Lateral,strainA,strainL] = Def_GLUE(Im1,Im2,coeffs,{0,0});

    GLUE.Axial = Axial;
    GLUE.Lateral = Lateral;

end