x_2d = [1991 1591 1036 2388 2924 2824; 590 1199 732 863 975 1696]; % 2 x N
X_3d = [2 6 3 3 2 6; 0 0 0 2 6 7; 2 4 8 0 0 0]; % 3 x N

% The above are coordinates of N(=6) corresponding points in the 3d world, 
% and the 2d image, w.r.to arbitrary origin and scale...
% Arbitrary origin and scale do not affect our calculations since
% the P(3 x 4) matrix can handle both these things...

if size(x_2d,1) ~= 2
    error('Points not 2d');
end
if size(X_3d,1) ~= 3
    error('Points not 3d');
end
if size(x_2d,2) ~= size(X_3d,2)
    error('Num of 2d and 3d points not the same');
end
N = size(x_2d,2);
if N < 6
    error('Num of points less than 6');
end

one_row = ones(1,N);
x_2d_homo = [x_2d; one_row];
X_3d_homo = [X_3d; one_row];

[x_2d_homo_norm, T] = normalisepts(x_2d_homo);
[X_3d_homo_norm, U] = normalisepts(X_3d_homo);

% Computing Matrix P
M = zeros(2*N, 12);
for ind = 1:N
    point = X_3d_homo_norm(:,ind)';
    M(2*ind-1,1:4) = (-1)*point;
    M(2*ind-1,9:12) = x_2d_homo_norm(1,ind)*point;
    M(2*ind,5:8) = (-1)*point;
    M(2*ind,9:12) = x_2d_homo_norm(2,ind)*point;
end

[~,~,V] = svd(M, 'econ'); % V will be 12x12
p = V(:,12); % 12th eigenvector is to be chosen
% p = p/norm(p);
Phat = reshape(p, [4,3])';

P = (T\Phat)*U; % inv(T)*Phat*U

H = P(:,1:3);
h = P(:,4);

% Recovering individual components
X0 = (-1)*(H\h);
[Q_,R_] = qr(inv(H));
R = Q_';
K = inv(R_);

% Reconstruction
x_2d_recons_homo = P*X_3d_homo;
coord_x = x_2d_recons_homo(1,:)./x_2d_recons_homo(3,:);
coord_y = x_2d_recons_homo(2,:)./x_2d_recons_homo(3,:);
x_2d_recons = [coord_x; coord_y];

SE_x = ( x_2d(1,:) - x_2d_recons(1,:) ).^2;
SE_y = ( x_2d(2,:) - x_2d_recons(2,:) ).^2;
SE = SE_x + SE_y;
RMSE = sqrt(mean(SE(:)));