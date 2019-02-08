rng(0);

%% Monument
% Loading and displaying the images
clear;
base_dir = '../input/monument/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    im{i} = imread(strcat(base_dir, files(i).name));
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 2, 0.05);
toc;
figure; imshow(p_im);
title('Stitched panaroma');

%% Hill
% Loading and displaying the images
clear;
base_dir = '../input/hill/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    im{i} = imread(strcat(base_dir, files(i).name));
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');

%% Ledge
% Loading and displaying the images
clear;
base_dir = '../input/ledge/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    im{i} = imread(strcat(base_dir, files(i).name));
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');

%% Pier
% Loading and displaying the images
clear;
base_dir = '../input/pier/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    im{i} = imread(strcat(base_dir, files(i).name));
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');

%% camera_2
% Loading and displaying the images
clear;
base_dir = '../input/camera_2/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    ii = imread(strcat(base_dir, files(i).name));
    im{i} = ii(1:4:end,1:4:end,:); % downsample because images are too big
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 1, 5);
toc;
figure; imshow(p_im);
title('Stitched panaroma');

%% camera_3
% Loading and displaying the images
clear;
base_dir = '../input/camera_3/';
files = dir(strcat(base_dir,'*.jpg'));
n = length(files);
im = cell(1,n);
for i=1:n
    ii = imread(strcat(base_dir, files(i).name));
    im{i} = ii(1:3:end,1:3:end,:); % downsample because images are too big
    figure; imshow(im{i});
    title(strcat('Original image -', string(i))); 
end

% Creating Panaroma
tic;
p_im = createPanaroma(im, 2, 5);
toc;
figure; imshow(p_im);
title('Stitched panaroma');