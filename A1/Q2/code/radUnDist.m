function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);
    [x, y] = meshgrid(1:n, 1:m);
    cx = m/2;
    cy = n/2;
    
    x = x - cx;
    y = y - cy;
    x = x/cx;
    y = y/cy;

    x_orig = x;
    y_orig = y;

    for k=1:nSteps
        r = sqrt(x.^2 + y.^2);
        h =  1 + k1*r + k2*r.^2;
        x =  x_orig./h;
        y =  y_orig./h;
    end

    x = x*cx + cx;
    y = y*cy + cy;
    
    imOut = interp2(imIn, x, y, 'cubic');
end