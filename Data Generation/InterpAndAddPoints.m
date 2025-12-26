function newVector = InterpAndAddPoints(originalVector,pointsNeeded)

    % This function takes originalVector as an input, performs
    % interpolation, and changes the length of the vector to be of the
    % length needed. The reason this is reversed is because I originally 
    % fucked up making the displacement set, should remake it if time.
    originalVector = double(originalVector);
    x = linspace(1,length(originalVector),length(originalVector));
    newx = linspace(min(x),max(x),pointsNeeded);
    newVector = interp1(x,originalVector,newx,"spline");