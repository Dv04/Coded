% Reading and displaying an image
img = imread('path/to/image.jpg')
imshow(img)

% Image information
info = imfinfo('path/to/image.jpg')
filename = info.Filename
filesize = info.FileSize
width = info.Width
height = info.Height
bitdepth = info.BitDepth
colortype = info.ColorType
filemoddate = info.FileModDate

% Subplot and channel separation
subplot(2, 2, 1); imshow(img)
redChannel = img(:, :, 1)
greenChannel = img(:, :, 2)
blueChannel = img(:, :, 3)
subplot(2, 2, 2); imshow(redChannel)
subplot(2, 2, 3); imshow(greenChannel)
subplot(2, 2, 4); imshow(blueChannel)

% RGB to Grayscale
grayImage = rgb2gray(img)
imshow(grayImage)
