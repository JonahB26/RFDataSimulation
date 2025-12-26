classdef QA
    methods (Static)
        function viewBmode(IQIMG,xL,zax)
            Idb = bmode(IQIMG, 50);
            % Display the B-mode image
            figure;
            imagesc(xL*1e3, zax*1e3, Idb); axis ij image; colormap gray; colorbar;
            xlabel('Lateral (mm)'); ylabel('Depth (mm)'); title('B-mode Image');
        end
        % function report(IQIMG, ENV, xL, zax, Itex256)
        %     assert(~isreal(IQIMG),'IQIMG must be complex.');
        %     ENVn = ENV / max(ENV(:)+eps);
        % 
        %     Itex = im2double(Itex256);
        %     Itex = imresize(Itex,[size(ENV,1) size(ENV,2)],'nearest');
        %     th   = graythresh(Itex);
        %     Mask = imfill(bwareaopen(Itex >= th, 50), 'holes');
        % 
        %     in_vals = ENVn(Mask);
        %     bg_vals = ENVn(~Mask & isfinite(ENVn));
        % 
        %     mu_in = mean(in_vals); sd_in = std(in_vals);
        %     mu_bg = mean(bg_vals); sd_bg = std(bg_vals);
        %     CNR   = abs(mu_in-mu_bg)/sqrt(sd_in^2+sd_bg^2);
        %     ENLbg = (mu_bg/max(sd_bg,eps))^2;
        %     fprintf('CNR=%.3f, ENL_bg≈%.2f\n', CNR, ENLbg);
        % 
        %     Idb = bmode(IQIMG, 50);
        % 
        %     figure('Name','Ultra QA');
        %     tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
        % 
        %     nexttile;
        %     imagesc(xL*1e3, zax*1e3, Idb); axis ij image; colormap gray; colorbar
        %     xlabel('mm'); ylabel('mm'); title('B-mode (50 dB)');
        % 
        %     nexttile;
        %     imagesc(xL*1e3, zax*1e3, ENVn); axis ij image; colormap gray
        %     hold on; contour(xL*1e3, zax*1e3, Mask,[0.5 0.5],'r','LineWidth',0.8);
        %     xlabel('mm'); ylabel('mm'); title('Envelope + mask');
        % 
        %     nexttile;
        %     plot(zax*1e3, ENVn(:,round(size(ENVn,2)/2))); grid on
        %     xlabel('Depth (mm)'); ylabel('|A|'); title('Mid A-line');
        % 
        %     nexttile;
        %     plot(xL*1e3, mean(ENVn,1)); grid on
        %     xlabel('Lateral (mm)'); ylabel('Mean |A|'); title('Mean lateral profile');
        % end

        function [hFig, savedPath] = report(IQIMG, ENV, xL, zax, Itex256, varargin)
            % REPORT  QA plots + optional save-to-image.
            % Usage:
            %   report(IQIMG,ENV,xL,zax,Itex256)                          % no save
            %   report(IQIMG,ENV,xL,zax,Itex256,'SavePath','qa.png')      % save PNG
            %   report(...,'Resolution',150,'CloseAfterSave',false)        % options
            %
            % Returns:
            %   hFig       : figure handle
            %   savedPath  : full path if saved, "" otherwise
            
                % ---- options ----
                p = inputParser; p.KeepUnmatched = true;
                addParameter(p,'SavePath',"",@(s)isstring(s)||ischar(s));
                addParameter(p,'Resolution',150,@(x)isnumeric(x)&&isscalar(x));
                addParameter(p,'CloseAfterSave',false,@(x)islogical(x)&&isscalar(x));
                parse(p,varargin{:});
                savePath = string(p.Results.SavePath);
                dpi      = p.Results.Resolution;
                doClose  = p.Results.CloseAfterSave;
            
                assert(~isreal(IQIMG),'IQIMG must be complex.');
                ENVn = ENV / max(ENV(:)+eps);
            
                Itex = im2double(Itex256);
                Itex = imresize(Itex,[size(ENV,1) size(ENV,2)],'nearest');
                th   = graythresh(Itex);
                Mask = imfill(bwareaopen(Itex >= th, 50), 'holes');
            
                in_vals = ENVn(Mask);
                bg_vals = ENVn(~Mask & isfinite(ENVn));
            
                mu_in = mean(in_vals); sd_in = std(in_vals);
                mu_bg = mean(bg_vals); sd_bg = std(bg_vals);
                CNR   = abs(mu_in-mu_bg)/sqrt(sd_in^2+sd_bg^2);
                ENLbg = (mu_bg/max(sd_bg,eps))^2;
                fprintf('CNR=%.3f, ENL_bg≈%.2f\n', CNR, ENLbg);
            
                Idb = bmode(IQIMG, 50);
            
                hFig = figure('Name','Ultra QA');
                tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
            
                nexttile;
                imagesc(xL*1e3, zax*1e3, Idb); axis ij image; colormap gray; colorbar
                xlabel('mm'); ylabel('mm'); title('B-mode (50 dB)');
            
                nexttile;
                imagesc(xL*1e3, zax*1e3, ENVn); axis ij image; colormap gray
                hold on; contour(xL*1e3, zax*1e3, Mask,[0.5 0.5],'r','LineWidth',0.8);
                xlabel('mm'); ylabel('mm'); title('Envelope + mask');
            
                nexttile;
                plot(zax*1e3, ENVn(:,round(size(ENVn,2)/2))); grid on
                xlabel('Depth (mm)'); ylabel('|A|'); title('Mid A-line');
            
                nexttile;
                plot(xL*1e3, mean(ENVn,1)); grid on
                xlabel('Lateral (mm)'); ylabel('Mean |A|'); title('Mean lateral profile');
            
                % ---- optional save ----
                savedPath = "";
                if strlength(savePath) > 0
                    % ensure folder exists
                    outDir = fileparts(savePath);
                    if outDir ~= "" && ~isfolder(outDir), mkdir(outDir); end
            
                    drawnow;  % make sure figure is rendered
                    try
                        exportgraphics(hFig, savePath, 'Resolution', dpi);   % R2020a+
                    catch
                        % robust fallback if exportgraphics is unavailable
                        try
                            frm = getframe(hFig); imwrite(frm.cdata, savePath);
                        catch
                            % last resort
                            print(hFig, '-dpng', sprintf('-r%d',dpi), savePath);
                        end
                    end
                    if isfile(savePath)
                        savedPath = savePath;
                        fprintf('Saved QA image → %s\n', savePath);
                        if doClose, close(hFig); end
                    else
                        warning('QA image not written: %s', savePath);
                    end
                end
            end

        function R = elastoSanity(IQ1, IQ2, xL, zax, param, Uz_expected_mm, doPlot)
        % Quick QA for elastography pair.
        % IQ1,IQ2: complex beamformed (Nax x Nlat)
        % xL, zax: axes (meters)
        % param: struct with fc (Hz), c0 (m/s)
        % Uz_expected_mm: optional ground-truth axial disp (mm) (scalar or map)
        % doPlot: optional logical
        
            if nargin < 7, doPlot = true; end
            R = struct();
        
            % 0) Guard rails
            assert(~isreal(IQ1) && ~isreal(IQ2), 'IQ must be complex.');
            assert(all(size(IQ1)==size(IQ2)), 'IQ frames must have equal size.');
            [Nax,Nlat] = size(IQ1);
        
            bad = ~isfinite(IQ1) | ~isfinite(IQ2);
            R.nan_frac = nnz(bad)/numel(bad);
        
            % 1) Normalized cross-correlation over small windows (decorrelation)
            Wz = 32; Wx = 8;   % window (adjust to taste)
            NCCs = zeros(Nax,Nlat);
            for c = 1:Wx:Nlat
                c2 = min(c+Wx-1, Nlat);
                for r = 1:Wz:Nax
                    r2 = min(r+Wz-1, Nax);
                    A = IQ1(r:r2, c:c2); B = IQ2(r:r2, c:c2);
                    a = A(:) - mean(A(:)); b = B(:) - mean(B(:));
                    NCCs(r:r2,c:c2) = real((a'*conj(b)) / (norm(a)*norm(b) + eps));
                end
            end
            R.ncc_mean = mean(NCCs(:),'omitnan');
            R.ncc_p10  = prctile(NCCs(:),10);
        
            % 2) Phase-based global axial shift (bulk) estimate
            % phase change Δφ ≈ -4π fc / c0 * Δz
            dphi = angle(IQ2 .* conj(IQ1));
            phi_med = median(dphi(:),'omitnan');
            dz_bulk = -phi_med * param.c0 / (4*pi*param.fc); % meters
            R.bulk_axial_shift_mm = 1e3 * dz_bulk;
        
            if exist('Uz_expected_mm','var') && ~isempty(Uz_expected_mm)
                if isscalar(Uz_expected_mm)
                    R.uz_expected_mm = Uz_expected_mm;
                    R.bulk_bias_mm   = R.bulk_axial_shift_mm - Uz_expected_mm;
                else
                    % map provided; compare to its median inside valid depth
                    uz_med = median(Uz_expected_mm(:),'omitnan');
                    R.uz_expected_mm = uz_med;
                    R.bulk_bias_mm   = R.bulk_axial_shift_mm - uz_med;
                end
            end
        
            % 3) Edge-column energy (detect "dead" sides)
            env1 = abs(IQ1); env2 = abs(IQ2);
            leftE  = mean(env1(:,1:round(Nlat*0.05)),'all') / mean(env1,'all');
            rightE = mean(env1(:,end-round(Nlat*0.05):end),'all') / mean(env1,'all');
            R.edge_energy = [leftE rightE];
        
            % 4) Light verdicts
            R.ok_ncc   = R.ncc_mean > 0.90;        % tweak thresholds to your liking
            R.ok_phase = abs(R.bulk_axial_shift_mm) < 5;   % sanity (<5 mm bulk)
            R.ok_edges = all(R.edge_energy > 0.25);
        
            if doPlot
                figure('Name','Elasto QA'); tiledlayout(2,2,'padding','compact','tilespacing','compact');
                nexttile; imagesc(xL*1e3,zax*1e3, 20*log10(abs(IQ1)/max(abs(IQ1(:)))+eps)); axis ij image; colormap gray
                title('Frame 1 (dB)'); xlabel('mm'); ylabel('mm');
                nexttile; imagesc(xL*1e3,zax*1e3, 20*log10(abs(IQ2)/max(abs(IQ2(:)))+eps)); axis ij image; colormap gray
                title('Frame 2 (dB)'); xlabel('mm'); ylabel('mm');
                nexttile; imagesc(xL*1e3,zax*1e3, NCCs,[0 1]); axis ij image; colorbar; title(sprintf('NCC (mean=%.2f, p10=%.2f)',R.ncc_mean,R.ncc_p10));
                xlabel('mm'); ylabel('mm');
                nexttile;
                bar([R.ncc_mean, R.ncc_p10, R.edge_energy]); 
                xticklabels({'NCC mean','NCC p10','Edge L','Edge R'});
                grid on; title(sprintf('Bulk shift=%.2f mm',R.bulk_axial_shift_mm));
            end
        end

    end
end
