function reconstruction_result = RunReconstruction(reconstruction_options,boundary_conditions,YM_Image,strainA, ...
    strainL,Disp_ax,Disp_lat)
% This function reconstructs the YM image based on stress values calculated by
% FEM.

% Set iteration to 1 before starting the loop. For display purposes. 
Iteration_number = 1;

% a and b are calculated based on the Rw numeric value defined in the
% ReconOpts class. Rw = a/b, the ratio for the weighted average YM
% calculation.
a = reconstruction_options.Rw/(reconstruction_options.Rw + 1);
b = 1/(reconstruction_options.Rw + 1);

%Set the axial and lateral resolutions
% axialresolution = 221;
% lateralresolution = 201;

axRes = size(YM_Image,1) + 1;
latRes = size(YM_Image,2) + 1;

% Start the loop and continue while converged = false.
while ~reconstruction_options.converged

    % material = Material(YM_Image, 0.45);
    % analysis_options = FEMOpts("cartesian", axialresolution, lateralresolution, "PLANE_STRESS"); 
    % 
    % result = RunFiniteElementAnalysis(analysis_options, material, boundary_conditions,false);

    ConfigureFEM();

    % Create material definition
    material = clib.FEM_Interface.Material_MATLAB;
    
    material.youngs_modulus = flatten(YM_Image);
    material.poissons_ratio = flatten(0.48*ones(size(YM_Image)));
    
    % Analysis options
    analysis_options = clib.FEM_Interface.AnalysisOptions;
    analysis_options.coordinate_system_type = "cartesian";
    analysis_options.element_type = "PLANE_STRAIN";
    analysis_options.axial_nodal_resolution = axRes;
    analysis_options.lateral_nodal_resolution = latRes;

    boundary_conditions = clib.FEM_Interface.BoundaryStruct;

    boundary_conditions.top_axial = ConvertPXToMM(Disp_ax(1,:));   
    boundary_conditions.bottom_axial = ConvertPXToMM(Disp_ax(end,:));

    boundary_conditions.top_lateral = ConvertPXToMM(Disp_lat(1,:));   
    boundary_conditions.bottom_lateral = ConvertPXToMM(Disp_lat(end,:));

    boundary_conditions.right_axial = ConvertPXToMM(Disp_ax(:,end)');
    boundary_conditions.left_axial = ConvertPXToMM(Disp_ax(:,1)');

    boundary_conditions.right_lateral = ConvertPXToMM(Disp_lat(:,end)');
    boundary_conditions.left_lateral = ConvertPXToMM(Disp_lat(:,1)');

    disp('Running FEA...')
    simresult = clib.FEM_Interface.RunFiniteElementAnalysis(boundary_conditions,...
                                                      size(YM_Image)+1,...
                                                      material, analysis_options);
    assignin("base","simresult",simresult)
    assignin("base","material",material)
    % axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
    % lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);

    % axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
    % lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);

    axial_stress = reshape(simresult.axial_stress.double(),axRes-1, latRes-1);
    lateral_stress = reshape(simresult.lateral_stress.double(),axRes-1, latRes-1);

    axial_stress = double(axial_stress);
    lateral_stress = double(lateral_stress);
    assignin("base","lateral_stress",lateral_stress);
    assignin("base","axial_stress",axial_stress);


    % Update the YM and calculate weighted average(Using Plane Stress)
    v = reshape(material.poissons_ratio.double(),axRes-1, latRes-1);
    assignin("base","v",v);

    % YMA_reciprocal = abs(strainA./(result.axial_stress-v.*result.lateral_stress));
    % YML_reciprocal = abs(strainL./(result.lateral_stress-v.*result.axial_stress));

    YMA_reciprocal = abs(strainA./(axial_stress-v.*lateral_stress));
    YML_reciprocal = abs(strainL./(lateral_stress-v.*axial_stress));

    h = fspecial('average',reconstruction_options.filter_size); 
    YMA_reciprocal_filt = imfilter(YMA_reciprocal,h);
    YML_reciprocal_filt = imfilter(YML_reciprocal,h);

    YMA_reciprocal_filt = imgaussfilt(YMA_reciprocal_filt);
    YML_reciprocal_filt = imgaussfilt(YML_reciprocal_filt);

    YMA = 1./YMA_reciprocal_filt;
    YML = 1./YML_reciprocal_filt;

    assignin("base","YMA",YMA);
    assignin("base","YML",YML);

    YMA = YMA./max(YMA(:));
    YML = YML./max(YML(:));

   if strcmp(reconstruction_options.type, 'axial')
            YM = YMA;
    elseif strcmp(reconstruction_options.type, 'lateral')
            YM = YML;
    elseif strcmp(reconstruction_options.type, 'combined')
            YM = a*YMA + b*YML;
    else
            error('Invalid type')
   end

    %Debugging
    if reconstruction_options.debugFlag
        disp('Iteration Number: ')
        disp(Iteration_number)
        disp(['Mean Abs Difference: ', num2str(mean(abs(YM(:) - YM_Image(:))))])
        disp("meanYM: ")
        disp(mean(abs(YM(:))))
        disp("meanYM_Image:")
        disp(mean(abs(YM_Image(:))))
    end

    if mean(abs(YM(:) - YM_Image(:))) < reconstruction_options.tolerance
        reconstruction_options.converged = true;
    end

    %Reset YM_Image if not converged
    YM_Image = YM;
    Iteration_number = Iteration_number + 1;
end
assignin("base","YM_Image",YM_Image);
assignin("base","YM",YM);

% Display image
reconstruction_result = YM;

    %Visualize Reconstruction
    if reconstruction_options.visualize
        figure
        imshow(YM,[0,5*median(YM, 'all')])
        title('Elastography Image')
        % imshow(YM,[0,20 000])
        colorbar
    end
end