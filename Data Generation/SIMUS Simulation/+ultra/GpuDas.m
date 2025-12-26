classdef GpuDas
    % Vectorized GPU DAS: forms one column for a given lateral x.
    properties
        xeg_gpu  % 1xNe (single)
        zaxg_gpu % Naxx1 (single)
        apodg_gpu% Nex1 (single)
        Fnum single
        fs   single
        c0   single
    end

    methods
        function obj = GpuDas(xe, zax, apod, Fnum, fs, c0)
            obj.xeg_gpu   = gpuArray(single(xe));
            obj.zaxg_gpu  = gpuArray(single(zax));
            obj.apodg_gpu = gpuArray(single(apod));
            obj.Fnum = single(Fnum);
            obj.fs   = single(fs);
            obj.c0   = single(c0);
        end

        function col = beamformColumn(obj, IQg, tauTX, xl)
            Nt  = size(IQg,1);
            xlg = gpuArray(single(xl));
            dz  = sqrt( (obj.xeg_gpu - xlg).^2 + obj.zaxg_gpu.^2 );     % Nax x Ne
            tS  = single(tauTX) + 2*dz/obj.c0;                           % Nax x Ne

            s   = tS*obj.fs + 1;
            i0  = max(1, min(Nt-1, floor(s)));
            a   = s - i0;

            % double indexing on GPU is simplest & fast
            ne   = size(IQg,2);
            i0d  = gpuArray(double(i0));
            off  = gpuArray(double(0:ne-1)) * double(Nt);
            idx0 = i0d + off; idx1 = idx0 + 1;

            RF0 = IQg(idx0); RF1 = IQg(idx1);
            V   = (1 - a).*RF0 + a.*RF1;

            mask = abs(obj.xeg_gpu - xlg) <= (obj.zaxg_gpu / obj.Fnum);
            V    = V .* (obj.apodg_gpu.' .* mask);
            col  = sum(V, 2);   % complex, Naxx1
        end
    end
end
