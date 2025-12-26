classdef Progress
    % Lightweight console progress: prints "Line i/N (xx%) ETA"
    properties
        title string = "Running"
        N (1,1) double {mustBePositive} = 1
        minInterval (1,1) double {mustBePositive} = 0.5   % seconds
        t0   uint64     % << must be uint64 for toc(token)
        t_last uint64   % << must be uint64 for toc(token)
    end
    methods
        function obj = Progress(title, N, varargin)
            if nargin>=1, obj.title = string(title); end
            if nargin>=2, obj.N = N; end
            p = inputParser;
            addParameter(p,'Hz',[],@(x)isnumeric(x)&&isscalar(x)&&x>0);
            addParameter(p,'MinInterval',0.5);
            parse(p,varargin{:});
            if ~isempty(p.Results.Hz)
                obj.minInterval = 1/p.Results.Hz;
            else
                obj.minInterval = p.Results.MinInterval;
            end
            obj.t0 = tic;                % uint64 token
            obj.t_last = tic;            % uint64 token
            fprintf('%s: starting...\n', obj.title);
        end

        function step(obj, i)
            if i==1 || toc(obj.t_last) >= obj.minInterval || i==obj.N
                frac = i/obj.N;
                t_elapsed = toc(obj.t0);                     % uses uint64 token
                eta = (t_elapsed/max(frac,eps)) - t_elapsed;
                fprintf(1, '\r%s: line %4d / %4d  (%3.0f%%)  elapsed %6.1fs  ETA %5.1fs', ...
                        obj.title, i, obj.N, 100*frac, t_elapsed, max(eta,0));
                if i==obj.N, fprintf(1, '\n'); end
                drawnow limitrate nocallbacks               % make it show in Live Editor
                obj.t_last = tic;                           % reset token (uint64)
            end
        end

        function finish(obj)
            fprintf('\n%s: done in %.2fs\n', obj.title, toc(obj.t0));
        end
    end
end
