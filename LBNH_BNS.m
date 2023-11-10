clear all
close all

%% Read image
tic

LP = 1;             % Controls the cutoff frequency of the low-pass filter
HP = 4;             % Controls the cutoff frequency of the high-pass filter
threshold = 0.70;   % Threshold used to separate the noise and the high-frequency components of the foreground signal  

% Define image folder and names
indir_uni = './';
fnames_uni = 'sample uniform (for speckle plugin).tif';

info = imfinfo([indir_uni,'\',char(fnames_uni())]);            

%% Setting parameters
stck_sz = 1;          % Number of slices
w = info.Width;                         % get image width
h = info.Height;                        % get image height
sigmaBP = w/(10)*LP;	
kc = nearest(sigmaBP*0.18);             % cut-off frequency between hp and lp filter 
sigmaLP = kc*2/2.355;                   
lambda = nearest(w/(2*kc));             
if mod(lambda,2) == 0                   
    lambda = lambda+1;
else
end

h = h+2*lambda;                         % increase image size by lambda for padding evalutation 
w = w+2*lambda;                         % padding

%% Create high and low pass filters
lp = lpgauss(h,w,sigmaLP);

sigmaHP = w/(10)*HP;	
kc = nearest(sigmaHP*0.18);             % cut-off frequency between hp and lp filter 
sigmaHP = kc*2/2.355;                   

hp = hpgauss(h,w,sigmaHP);

%% Image preprocessing

unifimg = imread([indir_uni,'\',char(fnames_uni())]); 
sizeim = size(size(unifimg));
if sizeim(2) >= 3
sizem = size(unifimg);
if sizem(3) == 4
    unifimg = unifimg(:,:,1:3);
end
elseif sizeim(2) == 2
    unifimg(:,:,2) = unifimg(:,:,1);
    unifimg(:,:,3) = unifimg(:,:,1);
end
size1 = size(size(unifimg));
if size1(2) == 3
    unifimg = rgb2gray(unifimg);
end
unifimg = mat2gray(unifimg);
u = padarray((single(unifimg)),[lambda lambda],'symmetric');
uni = gather(u);


%% add noise (Optional)

% gaussian_mean = 0;             % Gaussian noise mean
% gaussian_std = 0.15;           % Standard deviation of Gaussian noise
% poisson_lambda = mean(uni(:)); % Poisson noise parameters
% 
% % Generate Gaussian noise
% gaussian_noise = gaussian_mean + gaussian_std * randn(size(uni));
% 
% % Generate Poisson noise
% poisson_noise = poissrnd(poisson_lambda, size(uni));
% 
% % Mixed Gaussian and Poisson noise
% 
% mixed_noise_image = double(uni) + gaussian_noise + 0.1*poisson_noise;
% uni = uni + mixed_noise_image;
% uni = mat2gray(uni);

%% Calculate high and low frequency components

Hi = real(ifft2(fft2(uni).*hp));
Lo = real(ifft2(fft2(uni).*lp));
Hi = Hi(lambda+1:end-lambda,lambda+1:end-lambda,:);
Lo = Lo(lambda+1:end-lambda,lambda+1:end-lambda,:);
uni = uni(lambda+1:end-lambda,lambda+1:end-lambda,:);

%% Denoise

signal_withnoise_mask = Hi>0;
ws = 2;     % window size in denoising
u2 = padarray(signal_withnoise_mask,[ws ws],'symmetric');  %Expanded image

noise = zeros(h-2*lambda,w-2*lambda);

[row, col] = size(u2);
for ii = 1+ws:1:row-ws
    for jj = 1+ws:1:col-ws
        if sum(sum(u2(ii-ws:ii+ws,jj-ws:jj+ws))) <= threshold*(ws^2+1)^2
      
            noise(ii-ws,jj-ws) = Hi(ii-ws,jj-ws);
      
        end
    end
end

noise(noise<0) = 0;

signal = uni - Lo - noise;
signal(signal<0) = 0;

signal_noise = uni - Lo;
signal_noise(signal_noise<0) = 0;

%% The error-cancelled signal is averaged based on the surrounding signals.

input_image = signal;

image_mean = mean(input_image(:));
image_std = std(double(input_image(:)));

% Calculate threshold
threshold = threshold*(ws^2+1)^2;%image_mean + 3 * image_std;
output_image = input_image;
[rows, cols] = size(signal);
% Iterate through each pixel in the image
for i = 2:(rows - 1)
    for j = 2:(cols - 1)
        % Get the grayscale values of the surrounding 8 adjacent pixels
        neighbors = input_image(i-1:i+1, j-1:j+1);
        neighbors(2,2) = 0;
    
        neighbor_mean = mean(neighbors(:));
        % If the gray value of the current pixel is less than the gray value of the surrounding 8 adjacent pixels
        if  input_image(i, j) < neighbor_mean
            % Replace the gray value of the current pixel with the mean
            output_image(i, j) = neighbor_mean;
        end
    end
end
signal = output_image;

toc
%% Show

figure,imshow(uni),colormap('hot'),title('WideFiled')
%figure,imshow(Lo),colormap('hot'),title('Background')
%figure,imshow(noise,[]),colormap('hot'),title('Noise')
%figure,imshow(signal_noise,[]),colormap('hot'),title('Signal with noise');
figure,imshow(signal,[]),colormap('hot'),title('BackgroundfreeImage');
GT= imread('.\sample uniform (for speckle plugin)_HiLo.tif');

%% Calculate PSNR and SSIM

X=double(mat2gray(uni)*255);
X=double(mat2gray(signal)*255);
Y=double(mat2gray(GT)*255);
A = Y-X;
B = X.*Y;
MSE = sum(A(:).*A(:))/numel(Y);
PSNR = 10*log10(255^2/MSE);
SSIM = ssim_cal(X,Y);

display(PSNR);
display(SSIM);



%% SSIM
function re=SSIM_cal(X,Y) 
        X = normalize01(X)*255;
        Y = normalize01(Y)*255;

        X=double(X);
        Y=double(Y);

        ux=mean(mean(X));
        uy=mean(mean(Y));

        sigma2x=mean(mean((X-ux).^2));
        sigma2y=mean(mean((Y-uy).^2));   
        sigmaxy=mean(mean((X-ux).*(Y-uy)));

        k1=0.01;
        k2=0.03;
        L=255;
        c1=(k1*L)^2;
        c2=(k2*L)^2;
        c3=c2/2;

        l=(2*ux*uy+c1)/(ux*ux+uy*uy+c1);
        c=(2*sqrt(sigma2x)*sqrt(sigma2y)+c2)/(sigma2x+sigma2y+c2);
        s=(sigmaxy+c3)/(sqrt(sigma2x)*sqrt(sigma2y)+c3);

        re=l*c*s;

end

function re = normalize01(img)
a = min(min(img));
b = max(max(img));
re = (img-a)./(b-a);
end

function [mssim, ssim_map] = ssim_cal(img1, img2)

% ========================================================================
% SSIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2009 Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Structural SIMilarity (SSIM) index between two images
%
% Please refer to the following paper and the website with suggested usage
%
% Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
% quality assessment: From error visibility to structural similarity,"
% IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,
% Apr. 2004.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
%
% Note: This program is different from ssim_index.m, where no automatic
% downsampling is performed. (downsampling was done in the above paper
% and was described as suggested usage in the above website.)
%
% Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size
%            depends on the window size and the downsampling factor.
%
%Basic Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim, ssim_map] = ssim(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim, ssim_map] = ssim(img1, img2, K, window, L);
%
%Visualize the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%========================================================================
K = [0.03, 0.03];
L = 100;
window = ones(100);
if (nargin < 2 | nargin > 5)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) | (N < 11))
     ssim_index = -Inf;
     ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);  %
   K(1) = 0.01;          % default settings
   K(2) = 0.03;          %
   L = 255;                                     %
end

if (nargin == 3)
   if ((M < 11) | (N < 11))
     ssim_index = -Inf;
     ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
       ssim_index = -Inf;
       ssim_map = -Inf;
       return;
      end
   else
     ssim_index = -Inf;
     ssim_map = -Inf;
     return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
     ssim_index = -Inf;
     ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
       ssim_index = -Inf;
       ssim_map = -Inf;
       return;
      end
   else
     ssim_index = -Inf;
     ssim_map = -Inf;
     return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
     ssim_index = -Inf;
     ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
       ssim_index = -Inf;
       ssim_map = -Inf;
       return;
      end
   else
     ssim_index = -Inf;
     ssim_map = -Inf;
     return;
   end
end

img1 = double(img1);
img2 = double(img2);

% automatic downsampling
f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter 
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');

      img1 = img1(1:f:end,1:f:end);
      img2 = img2(1:f:end,1:f:end);
  end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
  denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

return
end
