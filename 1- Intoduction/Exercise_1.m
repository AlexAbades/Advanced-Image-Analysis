%% Create a 2D Kernel and covolve the image 

% Read the image and convert units into double
I = mat2gray(double(imread("fibres_xcth.png")));

% Create a 2D Kernel
h = fspecial('average', 5);

% Convolve the image 
fil = conv2(I,h);
h_x = h(1,:);
h_y = h(:,1);

fil2 = conv2(h_y,h_x,I);

% Show images 
figure
subplot(1,3,1)
imshow(mat2gray(I))
title('Original Image')
subplot(1,3,2)
imshow(fil)
title('2D Convolved Image')
subplot(1,3,3)
imshow(fil2)
title('1D filter')

difference = sum(sum(mat2gray(fil2-fil)))

%%

%% Create a 2D Kernel and covolve the image 
clear all
% Read the image and convert units into double
I = mat2gray(imread("fibres_xcth.png"));

% Create a 2D Kernel
h = fspecial('average', 5);

% Convolve the image 
fil = imfilter(I,h);
h_x = h(1,:);
h_y = h(:,1);

fil2 = imfilter(I, h_x);
fil3 = imfilter(fil2,h_y);

% Show images 
figure
subplot(1,3,1)
imshow(I)
title('Original Image')
subplot(1,3,2)
imshow(fil)
title('2D Convolved Image')
subplot(1,3,3)
imshow(fil3)
title('1D filter')

