function transducer_list = genTransducers()
    %This scripts generates the list of
    %transducers to be used in Field II
    
    % This transducer will is the default transducer.
    transducer_1 = struct();
    
        transducer_1.central_frequency = 5e6;                %  Transducer center frequency [Hz]
        transducer_1.sampling_frequency = 100e6;                %  Sampling frequency [Hz]
        transducer_1.speed_of_sound = 1540;                  %  Speed of sound [m/s]
        transducer_1.lambda = transducer_1.speed_of_sound/transducer_1.central_frequency;             %  Wavelength [m]
        transducer_1.element_width = transducer_1.lambda;            %  Width of element
        transducer_1.element_height = 10/1000;   %  Height of element [m]
        transducer_1.kerf = 0.1/1000;          %  Kerf [m]
        transducer_1.focus = [0 0 50]/1000;     %  Fixed focal point [m]
        transducer_1.N_elements = 192;          %  Number of physical elements
        transducer_1.N_active = 64;             %  Number of active elements 
        transducer_1.focal_zones = [30:10:50]'/1000;
        transducer_1.transmit_focus = 50/1000;          %  Transmit focus
    
    
    % This transducer uses the L11-5V transducer parameters.
    transducer_2 = struct();
    
        transducer_2.central_frequency=7.6e6;                %  Transducer center frequency [Hz]
        transducer_2.sampling_frequency=transducer_2.central_frequency * 10;                %  Sampling frequency [Hz]
        transducer_2.speed_of_sound=1540;                  %  Speed of sound [m/s]
        transducer_2.lambda=transducer_2.speed_of_sound/transducer_2.central_frequency;             %  Wavelength [m]
        transducer_2.element_width=270e-6;            %  Width of element
        transducer_2.element_height=5e-3;   %  Height of element [m]
        transducer_2.kerf=30e-6;          %  Kerf [m]
        transducer_2.focus=[0 0 18]/1000;     %  Fixed focal point [m]
        transducer_2.N_elements=128;          %  Number of physical elements
        transducer_2.N_active=transducer_2.N_elements/4;             %  Number of active elements 
        transducer_2.focal_zones = [10:10:50]'/1000;
        transducer_2.transmit_focus = 18/1000;          %  Transmit focus
    
    % This transducer uses the L12-3V transducer parameters.
    transducer_3 = struct();
    
        transducer_3.central_frequency=7.5e6;                %  Transducer center frequency [Hz]
        transducer_3.sampling_frequency=transducer_3.central_frequency * 10;                %  Sampling frequency [Hz]
        transducer_3.speed_of_sound=1540;                  %  Speed of sound [m/s]
        transducer_3.lambda=transducer_3.speed_of_sound/transducer_3.central_frequency;             %  Wavelength [m]
        transducer_3.element_width=170e-6;            %  Width of element
        transducer_3.element_height=5e-3;   %  Height of element [m]
        transducer_3.kerf=30e-6;          %  Kerf [m]
        transducer_3.focus=[0 0 18]/1000;     %  Fixed focal point [m]
        transducer_3.N_elements=192;          %  Number of physical elements
        transducer_3.N_active=transducer_3.N_elements/4;             %  Number of active elements 
        transducer_3.focal_zones = [10:10:50]'/1000;
        transducer_3.transmit_focus = 18/1000;          %  Transmit focus
    
     transducer_list = {transducer_1,transducer_2,transducer_3}; % Store the transducer objects in this list.
    % save('/home/deeplearningtower/Documents/MATLAB/JonahCode/testData/MiscellaneousMatFiles/transducer_list.mat',"transducer_list")
