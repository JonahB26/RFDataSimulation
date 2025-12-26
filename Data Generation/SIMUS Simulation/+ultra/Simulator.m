classdef Simulator
    % Orchestrates focused TX per-line SIMUS + GPU DAS, with pruning.
    properties
        arr   ultra.LinearArray
        grid  ultra.ImagingGrid
        prn   ultra.Pruner
        das   ultra.GpuDas
    end

    methods
        function obj = Simulator(arr, grid, prn)
            obj.arr = arr;
            obj.grid= grid;
            if isempty(obj.arr.Nt), obj.arr = obj.arr.trimNtToDepth(grid.zax(end)); end
            apod = hanning(obj.arr.Nelements);
            obj.das = ultra.GpuDas(arr.xe, grid.zax, apod, grid.Fnum, arr.fs, arr.c0);
            if nargin<3 || isempty(prn), obj.prn = ultra.Pruner(); else, obj.prn = prn; end
        end

        function tauTX = focusDelays(obj, xl)
            dtx   = sqrt((obj.arr.xe - xl).^2 + obj.grid.zf^2);
            tauTX = (max(dtx) - dtx) / obj.arr.c0;
        end

        function [RFCHAN, IQIMG, ENV, Idb, info] = generateFrame(obj, xs, zs, RC, opt)
            % ---- robust defaults for opt ----
            if nargin < 5 || ~isstruct(opt), opt = struct(); end
            if ~isfield(opt,'ProgressHz'), opt.ProgressHz = 0; end   % 0 = no progress
            if ~isfield(opt,'WaitBar'),    opt.WaitBar    = false; end
        
            % (optional) start progress
            if opt.ProgressHz > 0
                prog = ultra.Progress("Focused frame", obj.grid.Nlat, 'Hz', opt.ProgressHz);
            else
                prog = [];
            end
        
            lat_span = max(xs) - min(xs);
            oc = obj.prn.effectiveCoeff(obj.arr.pitch, lat_span);
        
            IQgImg = gpuArray(complex(zeros(obj.grid.Nax, obj.grid.Nlat, 'single')));
            ENVg   = gpuArray(zeros(  obj.grid.Nax, obj.grid.Nlat, 'single'));
        
            t0 = tic;
            RFgImg = gpuArray(zeros(obj.grid.Nax,obj.grid.Nlat,'single'));
            RFCHAN = gpuArray(zeros(obj.grid.Nax,obj.grid.Nlat,'single'));
            for il = 1:obj.grid.Nlat
                xl    = obj.grid.xL(il);
                tauTX = obj.focusDelays(xl);
        
                [xsL, zsL, RCL] = obj.prn.apply(xs, zs, RC, xl, oc, obj.grid.zax(1), obj.grid.zax(end), lat_span);
        
                % IMPORTANT: pass a struct to simus
                RF = simus(xsL, zsL, RCL, tauTX, obj.arr.toParamStruct(), opt);
                RF_raw = gather(RF);
                RFCHAN(:, il) = single(RF_raw);
        
                IQ = single(rf2iq(RF, obj.arr.fs, obj.arr.fc));
                IQg= gpuArray(IQ);
        
                col = obj.das.beamformColumn(IQg, tauTX, xl);
                IQgImg(:, il) = col;
                ENVg(:,  il)  = abs(col);

                t_ax  = gpuArray((0:obj.grid.Nax-1)') / obj.arr.fs;      % Nax x 1 (s)
                RFcol = real( col .* exp(1j*2*pi*obj.arr.fc .* t_ax) );  % Nax x 1, REAL
                RFgImg(:, il) = single(RFcol);
        
                if ~isempty(prog), prog.step(il); end   % progress tick (if enabled)
            end
            if ~isempty(prog), prog.finish(); end
        
            t_elapsed = toc(t0);
        
            IQIMG = gather(IQgImg); ENV = gather(ENVg); RFIMG = gather(RFgImg);
            Idb   = bmode(IQIMG, 50);
        
            info = struct('elapsed_s',t_elapsed, 'Nax',obj.grid.Nax, 'Nlat',obj.grid.Nlat, ...
                          'zf',obj.grid.zf, 'Fnum',obj.grid.Fnum, ...
                          'optimization_coeff',oc,'lat_span',lat_span);
        end

    end
end
