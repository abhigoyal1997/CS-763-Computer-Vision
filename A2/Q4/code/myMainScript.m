rng(0);

%% Loading and displaying the images (Monument)

tic;
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
p_im = createPanaroma(im, 2, 0.05);
toc;
figure; imshow(p_im);
title('Stitched panaroma');
clear;

%% Loading and displaying the images (Hill)

tic;
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
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');
clear;

%% Loading and displaying the images (Legde)

tic;
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
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');
clear;

%% Loading and displaying the images (Pier)

tic;
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
p_im = createPanaroma(im, 2, 0.005);
toc;
figure; imshow(p_im);
title('Stitched panaroma');
clear;