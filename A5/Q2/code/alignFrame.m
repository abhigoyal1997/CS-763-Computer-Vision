function [ F_n ] = alignFrame(F,tx,ty,theta)

H = [cos(theta) -sin(theta) tx; sin(theta) cos(theta) ty; 0 0 1];
[m,n,~] = size(F);
p = combvec(1:n,1:m);
p(3,:) = 1;

rp = H\p;
rp = round(rp(1:2,:)./rp(3,:))';
F_n = zeros(size(F));

for i=1:m
    for j=1:n
        k = (i-1)*n+j;
        if all(rp(k,:) > 0) && all(rp(k,:) <= [n,m])
            F_n(i,j,:) = F(rp(k,2),rp(k,1),:);
        end
    end
end

end

