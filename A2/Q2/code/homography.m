function [ H ] = homography( p1, p2 )
    [n,d] = size(p1);
    X1 = [p1 ones(n,1)]; % Points in homogenous co-ordinate system
    M = [-X1 zeros(n,d+1) p2(:,1).*X1; zeros(n,d+1) -X1 p2(:,2).*X1]; % M matrix in Mx = 0
    [~,~,V] = svd(M); % SVD to find argmin_x Mx
    H = reshape(V(:,end),3,3)'; % 2D homography matrix
end

