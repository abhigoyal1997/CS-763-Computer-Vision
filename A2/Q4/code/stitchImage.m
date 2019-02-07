function dest = stitchImage(src,dest,H)
    % Padding for size correction (to avoid cropping)
    [m,n,~] = size(src);
    [dm,dn,~] = size(dest);
    c = [1 n 1 n; 1 1 m m; 1 1 1 1];
    ch = H*c;
    ch = double(round(ch(1:2,:)./ch(3,:))');
    l_pad = -min(min(ch(:,1))-1,0);
    r_pad = max(max(ch(:,1))+1,dn)-dn;
    t_pad = -min(min(ch(:,2))-1,-1);
    b_pad = max(max(ch(:,2))+1,dm+1)-dm;
    
    dest = padarray(dest, [t_pad l_pad], 'pre');
    dest = padarray(dest, [b_pad r_pad], 'post');
    
    [m,n,~] = size(dest);
    
    % A correction to counter the translation caused by padding
    p = combvec(1:n,1:m);
    p(3,:) = 1;
    p(1,:) = p(1,:) - l_pad;
    p(2,:) = p(2,:) - t_pad;
    
    % Reverse warping
    rp = H\p;
    rp = round(rp(1:2,:)./rp(3,:))';
    
    for i=1:m
        for j=1:n
            k = (i-1)*n+j;
            if all(rp(k,:) > 0) && all(rp(k,:) <= [size(src,2),size(src,1)])
                dest(i,j,:) = max(src(rp(k,2),rp(k,1),:), dest(i,j,:));
            end
        end
    end    
end