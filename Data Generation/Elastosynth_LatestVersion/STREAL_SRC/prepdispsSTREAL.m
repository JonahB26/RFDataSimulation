function [Disp_ax,Disp_lat,strainA,strainL,strainS] = prepdispsSTREAL(axial_disp, lateral_disp)


    % Define the Kernel
    kernel = [0 0 0; -1 0 1; 0 0 0];
    kernel = kernel./sum(abs(kernel),'all');
    % Construct the ktensor for plane strain
    load("ClinicalData_KTensor.mat")
    
    kmat = zeros(218,198,3,6);

    for(i = 1:3)
        for(j = 1:6)
            kmat(:,:,i,j) = imresize(ktensor(:,:,i,j),[218,198]);
        end
    end

    % % Construct the ktensor for plane strain
    % kmat = zeros(218,198,3,6);
    % kmat(:,:,:,1) = 1;
    % kmat(:,:,:,2) = -1;
    
    % Optimization Params for STREAL
    DECAY = 1.001;
    TOLERANCE_BEFORE_EXIT = 1;
    MIN_IMPROVEMENT = 1.10;
    THRESHOLD_OF_IMPROVEMENT = 0.70;    
    
    % Aspect Ratio of Phantoms
    L = 20;
    D = 20;
    
    % Target Resolution
    target_dims = [220,200];
    
    % addpath("C:\Users\MattC\OneDrive\Masters\STREAL2\Full Algorithm");
    % addpath("C:\Users\MattC\OneDrive\Masters\STREAL2\Full Algorithm\Compatibility_Src");
    % addpath("C:\Users\MattC\OneDrive\Masters\STREAL2\Full Algorithm\Incompressibility_Src");

    ux = imresize(axial_disp,[221,201],'bicubic');
    uy = imresize(lateral_disp,[221,201],'bicubic');

    tic
    [strainA, strainL, Disp_ax, Disp_lat] = RunSTREAL(ux, uy, kernel, target_dims, kmat,...
                               L, D, DECAY, TOLERANCE_BEFORE_EXIT,...
                               MIN_IMPROVEMENT, THRESHOLD_OF_IMPROVEMENT);
    time = toc;

    strainL = strainL(2:end-1,2:end-1);
    strainA = strainA(2:end-1,2:end-1);

    Disp_ax = Disp_ax(3:end-1,3:end-1);
    Disp_lat = Disp_lat(3:end-1,3:end-1);

    Disp_ax = ConvertPXToMM(Disp_ax);
    % Disp_lat = ConvertPXToMM(Disp_lat);

    Disp_ax = -1*(max(Disp_ax(:)) + min(Disp_ax(:)) - Disp_ax);
    % Disp_lat = max(Disp_ax(:)) + min(Disp_ax(:)) - Disp_ax;

    strainA = imresize(strainA,[220,200],'bicubic');
    strainL = imresize(strainL,[220,200],'bicubic');
    Disp_ax = imresize(Disp_ax,[221,201],'bicubic');
    Disp_lat = imresize(Disp_lat,[221,201],'bicubic');

    t1 = conv2(Disp_ax,[-1 1],'valid');
    t2 = conv2(Disp_lat,[-1; 1],'valid');
    t1 = t1(1:end-1,:);
    t2 = t2(:,1:end-1);
    strainS = 0.5*(t1+t2);

    % time

end