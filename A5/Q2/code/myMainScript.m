%% Read Video & Setup Environment
clear
clc
close all hidden
[FileName,PathName] = uigetfile({'*.avi'; '*.mp4'},'Select shaky video file');

cd mmread
vid=mmread(strcat(PathName,FileName));
cd ..
s=vid.frames;

%% Extrating features and estimating motion trajectory

T = size(s,2);

mH = zeros(T-1,3,3);
H = zeros(T-1,3,3);
thres = 0.01;
f = waitbar(0,'Extrating features and estimating motion trajectory...');
for i=1:T-1
    [p1, p2] = featureMatch(s,i);
    mH(i,:,:) = ransacHomography(p1,p2,thres);
    if i>1
        H(i,:,:) = squeeze(H(i-1,:,:))*squeeze(mH(i,:,:));
    else
        H(i,:,:) = mH(i,:,:);
    end
    waitbar(i/(T-1));
end
close(f);

tx = H(:,1,end);
ty = H(:,2,end);
theta = real(asin(H(:,2,1)));

%% Mean filter
windowSize = 15;
b = (1/windowSize)*ones(1,windowSize);
a = 1;

mtx = filter(b,a,tx);
mty = filter(b,a,ty);
mtheta = filter(b,a,theta);

%% Stablizing video
txfinal = tx - mtx;
tyfinal = ty - mty;
thetafinal = theta - mtheta;

H(:,1,end) = txfinal;
H(:,2,end) = tyfinal;
H(:,1,1) = cos(thetafinal);
H(:,1,2) = -sin(thetafinal);
H(:,2,1) = sin(thetafinal);
H(:,2,2) = cos(thetafinal);

outV = s(1);
f = waitbar(0,'Stablizing video...');
for i=2:T
    F = s(i);
    F_i = alignFrame(F.cdata,squeeze(H(i-1,:,:)));
    outV(i) = struct('cdata',uint8(F_i));
    waitbar((i-1)/(T-1));
end
close(f);

%% Write Video
vfile=strcat(PathName,'combined_',FileName);
ff = VideoWriter(vfile);
ff.FrameRate = 30;
open(ff)

for i=1:T
    f1 = s(i).cdata;
    f2 = outV(i).cdata;
    vframe=cat(1,f1, f2);
    writeVideo(ff, vframe);
end
close(ff)

%% Display Video
figure
msgbox(strcat('Combined Video Written In ', vfile), 'Completed')
displayvideo(outV,0.01)
