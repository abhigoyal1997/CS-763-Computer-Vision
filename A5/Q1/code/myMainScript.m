%% Read Video & Setup Environment
clear
clc
close all hidden
[FileName,PathName] = uigetfile({'*.avi'; '*.mp4'},'Select shaky video file');

cd mmread
vid=mmread(strcat(PathName,FileName));
cd ..
s=vid.frames;

%% Your code here

T = size(s,2);
Tx = zeros(T-1,1);
Ty = zeros(T-1,1);
Theta = zeros(T-1,1);
thres = 1;
for i=1:T-1
    i
    [p1, p2] = featureMatch(s,i);
     [Tx(i), Ty(i), Theta(i)] = ransacHomography(p1,p2,thres);
end

%% Mean filter
windowSize = 5; 
b = (1/windowSize)*ones(1,windowSize);
a = 1;

Tx = cumsum(Tx);
Ty = cumsum(Ty);
Theta = cumsum(Theta);

Mtx = filter(b,a,Tx);
Mty = filter(b,a,Ty);
Mtheta = filter(b,a,Theta);

%% Smoothen video frames
txfinal = Tx - Mtx;
tyfinal = Ty- Mty;
thetafinal = Theta - Mtheta;

% txfinal = Tx - Mtx;
% tyfinal = Ty- Mty;
% thetafinal = Theta - Mtheta;

outV = s(1);
for i=2:T
    i
    F = s(i);
    F_i = alignFrame(F.cdata,txfinal(i-1),tyfinal(i-1),thetafinal(i-1));
    outV(i) = struct('cdata',uint8(F_i),'colormap',[]);
end

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
