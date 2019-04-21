function [ F_n ] = alignFrame(F,H)

[m,n,~] = size(F);
p = combvec(1:n,1:m);
p(3,:) = 1;

rp = H\p;
rp = round(rp(1:2,:)./rp(3,:))';
F_n = uint8(zeros(size(F)));

for i=1:m
    for j=1:n
        k = (i-1)*n+j;
        if all(rp(k,:) > 0) && all(rp(k,:) <= [n,m])
            F_n(i,j,:) = F(rp(k,2),rp(k,1),:);
        end
    end
end

end

