function AM2D_OUT = RunAM2D(Im1, Im2, params)

    % ---------------------------------------- %
    % ------------- DP Paerametes ------------ %
    % ---------------------------------------- %
    % IRF = [0 40];
    % IRF = [-40 0];
    % IA = [-1 1]; %Maximum allowed disparity in lateral D
    % alfa_DP = 0.15; % DP regularization weight
    % ---------------------------------------- %
    % ------------ 2D AM Paerametes ---------- %
    % ---------------------------------------- %
    % midA = 40;

    %clin params
% IRF = [-40 0];         % axial search window
% IA = [-4 4];           % lateral disparity range
% alfa_DP = 0.1;         % slightly looser DP smoothing
% 
% midA = 60;             % window size
% alfa = 6;              % axial reg
% beta = 12;              % lateral reg
% gamma = 0.04;         % lighter smoothing in lateral
% T = 0.2;              % IRLS threshold

%FIELD II PARAMS
    IRF = [-10 30];         % axial search window
IA = [-1 1];           % lateral disparity range
alfa_DP = 0.15;         % slightly looser DP smoothing

midA = 23;             % window size
alfa = 10;              % axial reg
beta = 20;              % lateral reg
gamma = 0.005;         % lighter smoothing in lateral
T = 0.1;              % IRLS threshold

    coeffs{5} = midA;
    % alfa = 10; %axial regularization, PREVIOUSLY 5
    % alfa = 5;
    % alfa = 3; %axial regularization
    coeffs{6} = alfa;
    % beta = 5; %lateral regularization, PREVIOUSLY 10
    % beta = 10;
    coeffs{7} = beta;
    % gamma = 0.01;%0.005; %lateral regularization 
    coeffs{8} = gamma;
    % T = .4; % threshold for IRLS
    % T = 0.1; % threshold for IRLS
    % T = 0.2;
    coeffs{10} = T;
    a_t = params.probe.a_t; % frequency dependent attenuation coefficient, in dB/cm/MHz
    f_0 = params.probe.fc; %ultrasound center freq. in MHz
    f_s = params.probe.fs;
    xx = calc_att (a_t, f_0, f_s); % to compensate for attenuation
    % 
    % % Tissue Mimicking Phantom Version
    % % ---------------------------------------- %
    % % ------------- DP Paerametes ------------ %
    % % ---------------------------------------- %
    % IRF = [-100 100];
    % IA = [4 4]; %Maximum allowed disparity in lateral D
    % alfa_DP = 0.15; % DP regularization weight
    % % ---------------------------------------- %
    % % ------------ 2D AM Paerametes ---------- %
    % % ---------------------------------------- %
    % midA = 450;
    % alfa = 5; %axial regularization
    % beta = 10; %lateral regularization
    % gamma = 0.005; %lateral regularization 
    % T = .2; % threshold for IRLS
    % a_t = params.probe.a_t; % frequency dependent attenuation coefficient, in dB/cm/MHz
    % f_0 = params.probe.fc; %ultrasound center freq. in MHz
    % f_s = params.probe.fs;
    % xx = calc_att (a_t, f_0, f_s); % to compensate for attenuation


    % IRF = [-35 35]; % Maximum allowed disparity in axial D
    % IA = [-4 4]; % Maximum allowed disparity in lateral D
    % midA = 50;
    % alfa = 5; %axial regularization
    % beta = 5; %lateral regularization
    % gamma = 0.005; %lateral regularization 
    % T = .2; % threshold for IRLS
    % alfa_DP = 0.15;
    % a_t = params.probe.a_t; % frequency dependent attenuation coefficient, in dB/cm/MHz
    % f_0 = params.probe.fc; %ultrasound center freq. in MHz
    % f_s = params.probe.fs; % sampling freq. in MHz
    % xx = calc_att (a_t, f_0, f_s); % to compensate for attenuation
    
    % [D1 D2 DPdisp] = AM2D(Im1, Im2, IRF, IA, midA, alfa_DP, alfa, beta, gamma, T, xx);
    [ax0, lat0, ~] = RCF_AM2D(Im1, Im2, IRF, IA, coeffs{5}, alfa_DP, coeffs{6}, coeffs{7}, coeffs{8}, coeffs{10}, xx);
    
    % lat0_denoised = medfilt2(lat0,[3 3]);

    AM2D_OUT.Axial = ax0;
    AM2D_OUT.Lateral = lat0;
    % AM2D_OUT.Lateral = lat0_denoised;

    % AM2D_OUT.Axial = D1;
    % AM2D_OUT.Lateral = D2;

end