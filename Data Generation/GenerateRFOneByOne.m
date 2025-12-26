function [parameter_table] = GenerateRFOneByOne(parameter_table, boundary_conditions, procedural_params, output_directory, speed_factor)
%GENERATERFONEBYONE This function generates the RF data one-by-one based on
%the table, going from the parameters all the way to the RF data at the
%end

    n_phantoms = height(parameter_table)

    progress_bar = waitbar(0/n_phantoms,"Generating Requested RF Data")

    for i = 1:n_phantoms

        phantom_params = parameter_table(i,:);

        if isfile(strcat(output_directory, "/", phantom_params.output_file))
            "THIS PHANTOM HAS ALREADY BEEN GENERATED"
        else
        
        base_message = strcat("Phantom #: ", string(i), ": ");
        waitbar(i/n_phantoms, progress_bar, strcat(base_message, "Generating Heterogeneity"))

        [~, ~, MetaData] = GenerateProceduralPhantom(phantom_params, procedural_params);

        waitbar(i/n_phantoms, progress_bar, strcat(base_message, "Running Finite Element Model"))

        figure
        imshow(MetaData.YM_hetero,[])
        colorbar

        MetaDataFEM = FEMRun(phantom_params, MetaData, boundary_conditions);

        load(parameter_table.transducer_file(i))

        % Calculate Image Options
        imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
        40/1000, 40/1000,10/1000, 10e4,100);
        imageopts.decimation_factor = 2;
        imageopts.axial_FOV = 60/1000;
        imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
        imageopts.slice_thickness = 10/1000;
        imageopts.speed_factor;
    
        MetaDataFEM.imageopts = imageopts;

        % Generate the Radiofrequency pairs

        field_init();
    
        waitbar(i/n_phantoms, progress_bar, strcat(base_message, "Generating RF Data Frames"))

        imageopts = MetaDataFEM.imageopts;

        D = imageopts.axial_FOV;
        L = imageopts.lateral_FOV;
        Z = imageopts.slice_thickness;

        [X,Y] = meshgrid(linspace(-L/2,L/2,MetaDataFEM.FEM_resolution(1)+1),linspace(0,D,MetaDataFEM.FEM_resolution(2)+1)+0.03);
        I = ones(220,200);
    
        [phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers);
        
        phantom = Phantom(phantom_positions, phantom_amplitudes);
        
        dispx = interp2(X,Y,MetaDataFEM.FEMresult.axial_disp,phantom_positions(:,1),phantom_positions(:,3));
        dispy = interp2(X,Y,MetaDataFEM.FEMresult.lateral_disp,phantom_positions(:,1),phantom_positions(:,3));

        displacements = zeros(imageopts.n_scatterers, 3);
        displacements(:,3) = dispx/1000;
        displacements(:,1) = dispy/1000;
        displacements(:,2) = parameter_table.OOP_displacement(i)/1000 * randn(imageopts.n_scatterers,1);

        tic
        
        [Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);

        output.metadata = MetaDataFEM;
        output.Frame1 = Frame1;
        output.Frame2 = Frame2;
        output.runtime = toc;


        % Hash and Save
        file_hex = DataHash(output, 'array','hex');

        filename = strcat(output_directory,"/", file_hex,".mat");
        save(filename, "output")
        parameter_table.output_file(i) = strcat(file_hex,".mat");


        writetable(parameter_table, strcat(output_directory, "/ParameterTable.csv"));

        end

    end
    
    close(progress_bar)

end

