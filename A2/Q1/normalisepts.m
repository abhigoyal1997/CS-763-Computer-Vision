function [newpts, A] = normalisepts(pts)
    % pts = dim x N array, where dim is the dimension of homogenous coords 
    dim = size(pts,1);
    N = size(pts,2);
    
%     Not needed because we ensure that pts has last coordinate = 1
%     for ind = 1:dim
%         pts(ind,:) = pts(ind,:)./pts(dim,:);
%     end
    
    center = mean(pts(1:(dim-1),:),2);
    newp = zeros(dim-1,N);
    for ind = 1:(dim-1)
        newp(ind,:) = pts(ind,:)-center(ind);
    end
    
    dist = newp(1,:).^2;
    for ind = 2:(dim-1)
        dist = dist + newp(ind,:).^2;
    end
    dist = sqrt(dist);
    meandist = mean(dist(:));
    scale = sqrt(dim-1)/meandist;
    
    A = zeros(dim);
    for ind = 1:(dim-1)
        A(ind,ind) = scale;
        A(ind,dim) = -scale*center(ind);
    end
    A(dim,dim) = 1;
    newpts = A*pts;