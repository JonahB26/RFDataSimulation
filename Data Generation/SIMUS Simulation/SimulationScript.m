clear; clc; gpuDevice([]);

% --- Probe/system ---
param = getparam('L11-5v');
param.c0 = 1540;
param.fs = max(4*param.fc, 30e6);

% % --- Texture + phantom (your workflow) ---
% Itex = load('P39-W2-S5-T.mat').TumorArea;
% Itex = imresize(Itex, [256 256]);
% 
% 
% 
% 
% bg_floor = 0.03; Gf = max(Itex, bg_floor);
% sigma = sqrt(2/pi); R = sigma * sqrt(-2*log(max(rand(size(Gf)), eps)));
% M = 0.2 + 0.8*R; S = Gf.*M; Itex = mat2gray(log(S+1e-6));
% 
% [xs,~,zs,RC] = genscat([3e-2 NaN], param, Itex, 0.5);
% [xs2,zs2,RC2] = rmscat(xs,zs,RC,zeros(1,param.Nelements),param);
% 
% % --- One frame (your native 96 lines) ---
% [IQ, ENV, Idb, info, sim] = ultra_generate_frame(param, xs2, zs2, RC2, ...
%     'Nax',2500,'Nlat',96,'xLFOVmm',[-20 20],'zmaxmm',30,'zf',20,'Fnum',1.7);

% fea_data_folder = '/home/deeplearningtower/Documents/JonahCode/FEAData';
fea_folder = uigetdir;
fea_files = dir(fullfile(fea_folder,'*.mat'));

for i = 1:5 %:length(fea_files)
    clearvars -except fea_folder fea_files i param
    % load results
    % Load results from the current file
    fea_data = load(fullfile(fea_folder, fea_files(i).name));
    Itex = fea_data.output.images.tumor_mask;
    cooper_mask = fea_data.output.images.cooper_mask;
    axial_disp = fea_data.output.disps.axial_disp;
    lateral_disp = fea_data.output.disps.lateral_disp;
    file_name = string(fea_files(i).name);
    file_name = erase(file_name,'.mat');
    file_name = strrep(file_name,'FeaData','SimusData');


    % --- Texture + phantom (your workflow) ---
    % Itex = load('P39-W2-S5-T.mat').TumorArea;
    % Itex = imresize(Itex, [256 256]);
    
    bg_floor = 0.03; Gf = max(Itex, bg_floor);
    sigma = sqrt(2/pi); R = sigma * sqrt(-2*log(max(rand(size(Gf)), eps)));
    M = 0.2 + 0.8*R; S = Gf.*M; 
    Itex = mat2gray(log(S+1e-6));              % base texture in [0,1]
    
    % (B) Inclusion mask you already have; if not, quick fallback:
    % inclusion_mask = your_existing_mask;   % preferred
    if ~exist('inclusion_mask','var') || isempty(inclusion_mask)
        inclusion_mask = imbinarize(Itex, graythresh(Itex));   % fallback guess
    end
    
    % (C) Priority: ligaments override inclusion on overlap
    inc_only = inclusion_mask & ~cooper_mask;
    
    % (D) Boost intensities so genscat places stronger/denser scatterers there
    %     (numbers are good starters; tweak to taste)
    gInc = 1.6;                % inclusion brighter than BG
    gLig = 2.2;                % ligaments brightest
    muI  = mean(Itex(:));
    
    Itex_boost = Itex;
    Itex_boost(inc_only)   = max(Itex_boost(inc_only),   min(1, gInc*muI + 0.05));
    Itex_boost(cooper_mask)= max(Itex_boost(cooper_mask),min(1, gLig*muI + 0.10));
    Itex_boost = min(max(Itex_boost,0),1);               % clip to [0,1]
    % ---------- END BLOCK ----------
    
    % Use boosted texture for scatterer generation (FAST & simple)
    [xs,~,zs,RC]   = genscat([3e-2 NaN], param, Itex_boost, 0.5);
    [xs2,zs2,RC2]  = rmscat(xs,zs,RC,zeros(1,param.Nelements),param);
    
    A = param.Nelements * param.pitch;          % physical aperture [m]
    margin_el = 12;                               % leave ~4 elements margin each side
    xmin = -(A/2 - margin_el*param.pitch);
    xmax =  (A/2 - margin_el*param.pitch);
    
    
    % --- One frame (your native 96 lines) ---
    [RF1, IQ1, ENV1, Idb1, info1, sim1] = ultra_generate_frame(param, xs2, zs2, RC2, ...
        'Nax',2500,'Nlat',96,'xLFOVmm',[xmin xmax]*1e3,'zmaxmm',30,'zf',20,'Fnum',2.0);

    png_file = strcat('SimusExamples/Images/Pre_',file_name,'.png');
    ultra.QA.report(IQ1,ENV1,sim1.grid.xL,sim1.grid.zax,Itex, 'SavePath',png_file,'CloseAfterSave',true)
    


    Ux = double(axial_disp)*1e-3;
    Uz = double(lateral_disp)*1e-3;
    
    % --- try BOTH orientations, pick the one that keeps the most in FOV
    xMin = min(sim1.grid.xL); xMax = max(sim1.grid.xL);
    zMin = sim1.grid.zax(1);  zMax = sim1.grid.zax(end);
    
    keep_best = 0; xsP_best = []; zsP_best = [];
    orient = '';
    
    for mode = 1:2
        if mode==1
            % A) rows=x, cols=z  (NDGRID(x,z) with sizes matching U directly)
            xv = linspace(xMin, xMax, size(Ux,1));
            zv = linspace(zMin, zMax, size(Ux,2));
            [Xg, Zg] = ndgrid(xv, zv);
            UxA = Ux; UzA = Uz;
            orient_try = 'NDGRID(x,z)';
        else
            % B) rows=z, cols=x  (typical FEM/meshgrid layout → transpose U)
            xv = linspace(xMin, xMax, size(Ux,2));
            zv = linspace(zMin, zMax, size(Ux,1));
            [Xg, Zg] = ndgrid(xv, zv);
            UxA = Ux.'; UzA = Uz.';        % transpose to match NDGRID(x,z)
            orient_try = 'transposed';
        end
    
        Fx = griddedInterpolant(Xg, Zg, UxA, 'linear', 'nearest');
        Fz = griddedInterpolant(Xg, Zg, UzA, 'linear', 'nearest');
        xsP = xs + Fx(xs, zs);
        zsP = zs + Fz(xs, zs);
    
        inFOV = (xsP>=xMin & xsP<=xMax & zsP>=zMin & zsP<=zMax);
        kept = nnz(inFOV);
        fprintf('FEA try %-12s → kept %d / %d (%.1f%%)\n', orient_try, kept, numel(xs), 100*kept/numel(xs));
    
        if kept > keep_best
            keep_best = kept; xsP_best = xsP(inFOV); zsP_best = zsP(inFOV);
            RC_best = RC(inFOV); orient = orient_try;
        else
            RC_best = RC;
        end
    end
    
    assert(keep_best>0, 'All post-scatterers left FOV; check units and FOV.');
    fprintf('Using orientation: %s\n', orient);

    [RF2, IQ2, ENV2, Idb2, info2,sim2] = ultra_generate_frame(param,xsP,zsP,RC_best, ...
        'Nax',2500,'Nlat',96,'xLFOVmm',[xmin xmax]*1e3,'zmaxmm',30,'zf',20,'Fnum',2.0);
    png_file = strcat('SimusExamples/Images/Post_',file_name,'.png');
    ultra.QA.report(IQ2,ENV2,sim2.grid.xL,sim2.grid.zax,Itex, 'SavePath',png_file,'CloseAfterSave',true)
    
    
    simus_output = struct();
    simus_output.Frame1 = RF1;
    simus_output.Frame2 = RF2;

    disp(size(Frame1))
    
    filename = strcat("SimusExamples/RFData/", file_name,".mat");
    save(filename, "simus_output")


%     % figure; imagesc(sim.grid.xL*100, sim.grid.zax*1000, Idb);
%     % axis ij image; colormap gray
%     % title(sprintf('Focused (Nlat=%d), time=%.2fs', sim.grid.Nlat, info.elapsed_s));
%     % xlabel('cm'); ylabel('mm');
% 
%     % --- QA (optional) ---
% 
%     % --- Elastography pair (replace with your real FEA fields) ---
%     % [Xg,Zg] = meshgrid(linspace(min(sim.grid.xL), max(sim.grid.xL), 257), ...
%     %                    linspace(sim.grid.zax(1), sim.grid.zax(end), 257));
% % xv = linspace(min(sim.grid.xL), max(sim.grid.xL), size(Ux,1));
% % zv = linspace(min(sim.grid.zax), max(sim.grid.zax), size(Ux,2));
% % dx = (xv(end)-xv(1)) / (numel(xv)-1);
% % dz = (zv(end)-zv(1)) / (numel(zv)-1);
% % [Xg,Zg] = ndgrid(xv, zv);
% % 
% % [xs_post, zs_post] = ultra.applyFEA(xs2, zs2, Xg, Zg, Ux, Uz, 'scale',[dx dz]);
% 
% % --- Elastography pair (replace with your real FEA fields) ---
% % [Xg,Zg] = meshgrid(linspace(min(sim.grid.xL), max(sim.grid.xL), 257), ...
% %                    linspace(sim.grid.zax(1), sim.grid.zax(end), 257));
% % Ux, Uz are 257x257 (currently in mm)  ⟵ convert to meters
% % Ux = double(Ux) * 1e-3;
% % Uz = double(Uz) * 1e-3;
% 
% % Build NDGRID over your *imaging FOV* (match U size!)
% xv = linspace(min(sim.grid.xL), max(sim.grid.xL), size(Ux,1));
% zv = linspace(sim.grid.zax(1), sim.grid.zax(end), size(Ux,2));
% [Xg, Zg] = ndgrid(xv, zv);   % << IMPORTANT: NDGRID, not meshgrid
% 
% % Ux = zeros(size(Xg)); Uz = zeros(size(Zg));  % (fill with FEA)
% % Ux = fea_data.output.disps.axial_disp;  
% % Uz = fea_data.output.disps.lateral_disp; 
% % Xg = Xg.'; Zg = Zg.'; Ux = Ux.'; Uz = Uz.';   % convert to NDGRID format
% [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,infoPair, sim2] = ...
%     ultra_generate_two_frames_with_fea(param, xs2, zs2, RC2, Xg, Zg, Ux, Uz, ...
%         'Nax',2500,'Nlat',10,'xLFOVmm',[xmin xmax],'zmaxmm',30,'zf',20,'Fnum',2.0);
% 
% 
%     % Ux = zeros(size(Xg)); Uz = zeros(size(Zg));  % (fill with FEA)
%     % Ux = axial_disp;  % Assign axial displacement to Ux for elastography
%     % Uz = lateral_disp; % Assign lateral displacement to Uz for elastography
%     % Xg = Xg.'; Zg = Zg.'; Ux = Ux.'; Uz = Uz.';   % convert to NDGRID format
% 
%     % [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,infoPair, sim2] = ...
%     %     ultra_generate_two_frames_with_fea(param, xs2, zs2, RC2, Xg, Zg, Ux, Uz, ...
%     %         'Nax',2500,'Nlat',10,'xLFOVmm',[xmin xmax],'zmaxmm',30,'zf',20,'Fnum',2.0);
%     %%
%     % ultra.QA.report(IQ1, ENV1, sim.grid.xL, sim.grid.zax, Itex);
%     % ultra.QA.report(IQ2, ENV2, sim.grid.xL, sim.grix.zax, Itex);
% 
%     % After generating the two frames IQ1, IQ2:
%     R = ultra.QA.elastoSanity(IQ1, IQ2, sim.grid.xL, sim.grid.zax, param, median(Uz(:))*1e3);
%     disp(R)
% 
% 
% 
%     figure;
%     subplot(121)
%     imagesc(xL*1e3, zax*1e3, Idb1); axis ij image; colormap gray; colorbar
%             xlabel('mm'); ylabel('mm'); title('B-mode (Pre) (50 dB)');
% 
%     subolot(122)
%     imagesc(xL*1e3, zax*1e3, Idb2); axis ij image; colormap gray; colorbar
%             xlabel('mm'); ylabel('mm'); title('B-mode (Post) (50 dB)');
    
    
    % 4) Send image only to your default ntfy topic
    if mod(i-1,20) == 0
        notifyMe(sprintf('Completed simulation for file %s', fea_files(i).name));
    end
end