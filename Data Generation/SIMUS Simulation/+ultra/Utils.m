classdef Utils
    methods (Static)
        function r = safeRange(x)
            r = max(x) - min(x);
        end
    end
end
