%% Input barbara image
im1 = double(imread('../input/barbara.png'));
%im1 = double(rgb2gray(imread('../input/flash1.jpg')));
im2 = double(imread('../input/negative_barbara.png'));
%im2 = double(rgb2gray(imread('../input/noflash1.jpg')));
% imshow(im2);

%% Modify moving image
[m, n] = size(im1);
new_im2 = imrotate(im2, 23.5, 'crop');
new_im2 = imtranslate(new_im2, [-3, 0],'FillValues', 0);
new_im2 = new_im2 + 8*rand(size(new_im2));
new_im2(new_im2<0) = 0;
new_im2(new_im2>255) = 255;
imshow(mat2gray(new_im2));


%% Get joint entropy
minAngle = -100;
mintx = -100;
minValue = 1000;
entropy = zeros(121,25);
for angle = -60:60
    for tx = -12:12
        newIm = moveImage(new_im2, angle, tx);
        entropy(angle+61,tx+13) = getJointEntropy(im1, newIm);
        if(entropy(angle+61,tx+13)<minValue)
            minAngle = angle;
            mintx = tx;
            minValue = entropy(angle+61,tx+13);
        end
    end
end

surf(entropy);
figure, imshow(mat2gray(entropy/max(entropy(:))));


        


