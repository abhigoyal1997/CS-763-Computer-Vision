function [ H ] = ransacHomography( x1, x2, thresh )
    k = 2;
    n = size(x1,1);
    max_iter = 3000;
    
    best_nc = 0;
    best_C = [];
    
    for i=1:max_iter
        fp = randperm(n,k);
        H = alignpoints(x1(fp,:), x2(fp,:));

        err = sre(H,x1,x2);
        C = find(err<=thresh);
        if size(C,1) > best_nc
            best_nc = size(C,1);
            best_C = C;
        end
    end
    
    H = alignpoints(x1(best_C,:), x2(best_C,:));
end

