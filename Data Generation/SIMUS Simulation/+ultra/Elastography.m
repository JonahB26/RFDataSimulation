% classdef Elastography
%     % Utilities for pre/post generation using FEA displacements.
%     methods (Static)
%         function [xs_post, zs_post] = applyFEA(xs, zs, Xg, Zg, Ux, Uz)
%             Fx = griddedInterpolant(Xg, Zg, Ux, 'linear','nearest');
%             Fz = griddedInterpolant(Xg, Zg, Uz, 'linear','nearest');
%             xs_post = xs + Fx(xs, zs);
%             zs_post = zs + Fz(xs, zs);
%         end
% 
%         function [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info] = ...
%                  twoFrames(sim, xs, zs, RC, Xg, Zg, Ux, Uz, opt)
%             % if nargin<9, opt = struct('WaitBar', false, 'ProgressHz',2); end
%             [xs_post, zs_post] = ultra.Elastography.applyFEA(xs, zs, Xg, Zg, Ux, Uz);
% 
%             xMin = min(sim.grid.xL); xMax = max(sim.grid.xL);
%             zMin = sim.grid.zax(1);  zMax = sim.grid.zax(end);
% 
%             inFOV = xs_post>=xMin & xs_post<=xMax & zs_post>=zMin & zs_post<=zMax;
%             fprintf('Post scatterers kept: %d / %d (%.1f%%)\n', nnz(inFOV), numel(xs_post), 100*mean(inFOV));
% 
%             xs_p = xs_post(inFOV);
%             zs_p = zs_post(inFOV);
%             RC_p = RC(inFOV);
% 
% 
%             [IQ1, ENV1, Idb1, info1] = sim.generateFrame(xs, zs,   RC, opt);
%             [IQ2, ENV2, Idb2, info2] = sim.generateFrame(xs_p, zs_p, RC_p, opt);
%             info = struct('pre',info1,'post',info2);
%             % IQ1 = [];
%             % IQ2 = [];
%             % ENV1 = [];
%             % ENV2 = [];
%             % Idb1 = [];
%             % Idb2 = [];
%             % info= [];
%         end
%     end
% end


classdef Elastography
% Utilities for pre/post generation using FEA displacements.
methods (Static)

    % --- FEA application (NDGRID + mm->m). Keeps your signature. ---
    function [xs_post, zs_post] = applyFEA(xs, zs, Xg, Zg, Ux, Uz, units_opt)
        % xs,zs in meters. Ux,Uz same size as Xg,Zg.
        % units_opt: 'mm' (default) or 'm'
        if nargin < 7 || isempty(units_opt), units_opt = 'mm'; end

        xs = double(xs); zs = double(zs);
        Xg = double(Xg); Zg = double(Zg);
        Ux = double(Ux); Uz = double(Uz);

        % Units: mm -> m (typical for your fields)
        if strcmpi(units_opt,'mm')
            Ux = Ux * 1e-3; Uz = Uz * 1e-3;
        end

        % Build interpolants (linear, clamp outside)
        Fx = griddedInterpolant(Xg, Zg, Ux, 'linear', 'nearest');
        Fz = griddedInterpolant(Xg, Zg, Uz, 'linear', 'nearest');

        xs_post = xs + Fx(xs, zs);
        zs_post = zs + Fz(xs, zs);
    end

    % --- Two-frame generator: rebuild NDGRID from sim.FOV, then filter FOV ---
    function [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info] = ...
        twoFrames(sim, xs, zs, RC, Xg, Zg, Ux, Uz, opt)

        % defaults (keeps your style)
        if nargin < 9 || isempty(opt), opt = struct; end
        if ~isfield(opt,'Units'),     opt.Units   = 'mm';  end     % Ux/Uz in mm by default
        if ~isfield(opt,'Verbose'),   opt.Verbose = true; end

        % ---------- PRE frame ----------
        [IQ1, ENV1, Idb1] = sim.generateFrame(xs, zs, RC);

        % ---------- Build NDGRID over imaging FOV to match U size ----------
        % (Ignore incoming Xg,Zg and rebuild; this is what fixed your 99% keep)
        Nu = size(Ux,1);  Nz = size(Ux,2);       % 257 x 257 for your fields
        xMin = min(sim.grid.xL);  xMax = max(sim.grid.xL);
        zMin = sim.grid.zax(1);   zMax = sim.grid.zax(end);
        xv = linspace(xMin, xMax, Nu);
        zv = linspace(zMin, zMax, Nz);
        [Xg_nd, Zg_nd] = ndgrid(xv, zv);

        % ---------- Apply FEA ----------
        [xs_post, zs_post] = ultra.Elastography.applyFEA( ...
                                xs, zs, Xg_nd, Zg_nd, Ux, Uz, opt.Units);

        % ---------- Keep only post-scatterers still in FOV ----------
        inFOV = (xs_post>=xMin & xs_post<=xMax & zs_post>=zMin & zs_post<=zMax);
        kept  = nnz(inFOV);
        if opt.Verbose
            fprintf('Post scatterers kept: %d / %d (%.1f%%)\n', ...
                    kept, numel(xs_post), 100*kept/numel(xs_post));
        end

        xs_p = xs_post(inFOV);  zs_p = zs_post(inFOV);  RC_p = RC(inFOV);

        % ---------- POST frame ----------
        if kept == 0
            warning('twoFrames:NoPostScatterers','All post-scatterers left FOV.');
            IQ2  = complex(zeros(size(IQ1),'like',IQ1));
            ENV2 = zeros(size(ENV1),'like',ENV1);
            Idb2 = zeros(size(Idb1),'like',Idb1);
        else
            [IQ2, ENV2, Idb2] = sim.generateFrame(xs_p, zs_p, RC_p);
        end

        % ---------- info ----------
        info = struct('pct_kept', 100*kept/max(1,numel(xs_post)), ...
                      'xFOV', [xMin xMax], 'zFOV', [zMin zMax], ...
                      'NuNz', [Nu Nz], 'Units', opt.Units);
    end
end
end
