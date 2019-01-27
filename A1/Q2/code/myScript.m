close all
clc
clear
im = double(imread('../input/chrysler.png'))./255;
figure
imshow(im)

% Test values. Do test for other values too
k1 = 0.1;
k2 = 0.01;
%%
imD = radDist(im, k1, k2);
figure
imshow(imD)

%%

% large values of parameters k1 and k2 need more number of steps. Though
% the original parameters needed only 2 steps
nSteps = 20; % Fill in the number of steps here
imU = radUnDist(imD, k1, k2, nSteps);
figure
imshow(imU)

