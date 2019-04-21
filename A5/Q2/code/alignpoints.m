function [ H ] = alignpoints( p1, p2 )
    
    c1 = mean(p1,1);
    c2 = mean(p2,1);
    

    p1 = p1 - c1;
    p2 = p2 - c2;
    
    [U,~,V] = svd(p2'*p1);
    R = V*U';
    if(det(R)<0)
        J = eye(size(U,1));
        J(end,end) = -1;
        R = V*J*U';

    end
    
    theta = acos(R(1,1));
    cdiff = c1'-R*c2';
    
    H = [R cdiff; 0 0 1];
end

