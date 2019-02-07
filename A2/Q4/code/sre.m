function err = sre(H,p1,p2)
    n = size(p1,1);
    X1 = [p1';ones(1,n)];
    X_h = H*X1;
    pts = (X_h(1:2,:)./X_h(3,:))';
    err = sum((p2-pts).^2,2);
end

