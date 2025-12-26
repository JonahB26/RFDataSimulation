function [RF_Data, Tstarts] = GenerateRFLinearArray(phantom, transducer, imageopts, optimization_coeff)

    f0=transducer.central_frequency;                %  Transducer center frequency [Hz]
    fs=transducer.sampling_frequency;                %  Sampling frequency [Hz]
    width=transducer.element_width;            %  Width of element
    element_height=transducer.element_height;   %  Height of element [m]
    kerf=transducer.kerf;          %  Kerf [m]
    focus=transducer.focus;     %  Fixed focal point [m]
    N_elements=transducer.N_elements;          %  Number of physical elements
    N_active=transducer.N_active;             %  Number of active elements 
    
    %  Set the sampling frequency
    
    set_sampling(fs);
    
    %  Generate aperture for emission
    
    xmit_aperture = xdc_linear_array (N_elements, width, element_height, kerf, 1, 10,focus);
    
    %  Set the impulse response and excitation of the xmit aperture
    
    impulse_response=sin(2*pi*f0*(0:1/fs:2/f0));
    impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
    xdc_impulse (xmit_aperture, impulse_response);
    
    excitation=sin(2*pi*f0*(0:1/fs:2/f0));
    xdc_excitation (xmit_aperture, excitation);
    
    %  Generate aperture for reception
    
    receive_aperture = xdc_linear_array (N_elements, width, element_height, kerf, 1, 10,focus);
    
    %  Set the impulse response for the receive aperture
    
    xdc_impulse (receive_aperture, impulse_response);
    
    %   Load the computer phantom
    
    %  Set the different focal zones for reception
    
    focal_zones=transducer.focal_zones;
    Nf=max(size(focal_zones));
    focus_times=(focal_zones-10/1000)/transducer.speed_of_sound;
    z_focus=transducer.transmit_focus;          %  Transmit focus
    
    %  Set the apodization
    
    apo=hanning(N_active)';
    
    %   Do linear array imaging
    image_width=imageopts.image_width;            %  Size of image sector
    d_x=imageopts.d_x;       %  Increment for image
    RF_Data = cell(imageopts.no_lines,1);
    Tstarts = zeros(imageopts.no_lines,1);
    
    % Do imaging line by line
    for i=[1:imageopts.no_lines]
    
      %  Test if the file for the line exist.
      %  Skip the simulation, if the line exits and
      %  go the next line. Else make the simulation
    
    
        %  Save a file to reserve the calculation
        
        %cmd=['save rf_data/rf_ln',num2str(i),'.mat i'];
        %eval(cmd);
        
        %  The the imaging direction
        
        x= -image_width/2 +(i-1)*d_x;
        
        %   Set the focus for this direction with the proper reference point
        
        xdc_center_focus (xmit_aperture, [x 0 0]);
        xdc_focus (xmit_aperture, 0, [x 0 z_focus]);
        xdc_center_focus (receive_aperture, [x 0 0]);
        xdc_focus (receive_aperture, focus_times, [x*ones(Nf,1), zeros(Nf,1), focal_zones]);
        
        %  Calculate the apodization 
        
        N_pre  = round(x/(width+kerf) + N_elements/2 - N_active/2);
        N_post = N_elements - N_pre - N_active;
        apo_vector=[zeros(1,N_pre) apo zeros(1,N_post)];
        xdc_apodization (xmit_aperture, 0, apo_vector);
        xdc_apodization (receive_aperture, 0, apo_vector);
        
        %   Calculate the received response

        phantom_positions = phantom.positions;
        phantom_amplitudes = phantom.amplitudes;

        disp(class(phantom_positions));
        disp(class(phantom_positions(:,1)));
        disp(isnumeric(phantom_positions(:,1)));
        
        % start_range = x - optimization_coeff*builtin('range',range(phantom_positions(:,1)));
        start_range = x - optimization_coeff*(max(phantom_positions(:,1)) - min(phantom_positions(:,1)));
        % end_range = x + optimization_coeff*builtin('range',range(phantom_positions(:,1)));
        end_range = x + optimization_coeff*(max(phantom_positions(:,1)) - min(phantom_positions(:,1)));
        
        valid = (phantom_positions(:,1) < end_range& phantom_positions(:,1) > start_range);
        truncated_positions = phantom_positions(valid,:);
        truncated_amplitudes = phantom_amplitudes(valid);
        
        [rf_data, tstart]=calc_scat(xmit_aperture, receive_aperture, truncated_positions, truncated_amplitudes);
        
        RF_Data{i} = rf_data;
        Tstarts(i) = tstart;
    end
    
    %   Free space for apertures
    
    xdc_free (xmit_aperture)
    xdc_free (receive_aperture)

end