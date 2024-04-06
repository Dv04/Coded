a = imread('cameraman.tif');
l1 = 255;
s1 = l1 - 1 - a;
subplot(1, 2, 1); imshow(a); title('Original');
subplot(1, 2, 2); imshow(s1); title('Gray Scale Image');

img = imread('cameraman.tif');
r2 = double(img);
c2 = 1;
l2 = 255;
s2 = c2 * log(1 + r2);
t2 = l2 / (c2 * log(1 + l2));
b2 = uint8(t2 * s2);
subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(b2); title('Logarithmic');

r3 = double(a);
c3 = 1;
l3 = 255;
g1 = 0.5;
s3 = c3 * (r3 .^ g1);
t3 = l3 / (c3 * (l3 .^ g1));
b3 = uint8(t3 * s3);
subplot(1, 2, 1); imshow(a); title('Original');
subplot(1, 2, 2); imshow(b3); title('Gamma');

r3 = double(a);
c3 = 1;
l3 = 255;
g1 = 1;
s3 = c3 * (r3 .^ g1);
t3 = l3 / (c3 * (l3 .^ g1));
b3 = uint8(t3 * s3);
subplot(1, 2, 1); imshow(a); title('Original');
subplot(1, 2, 2); imshow(b3); title('Gamma');

r3 = double(a);
c3 = 1;
l3 = 255;
g1 = 5;
s3 = c3 * (r3 .^ g1);
t3 = l3 / (c3 * (l3 .^ g1));
b3 = uint8(t3 * s3);
subplot(1, 2, 1); imshow(a); title('Original');
subplot(1, 2, 2); imshow(b3); title('Gamma');

% You need to provide the pixel coordinates where you want to see the neighborhood
a = magic(8);
b = input('Enter Row: ');
c = input('Enter col: ');

N4 = [a(b - 1, c), a(b, c + 1), a(b, c - 1), a(b + 1, c)];
N4

N8 = [a(b - 1, c - 1), a(b - 1, c), a(b - 1, c + 1), a(b, c - 1), a(b, c + 1), a(b + 1, c - 1), a(b + 1, c), a(b + 1, c + 1)];
N8

ND = [a(b - 1, c - 1), a(b - 1, c + 1), a(b + 1, c - 1), a(b + 1, c + 1)];
ND
% Define the neighborhood using the provided pixel coordinates

a = ones(40);
b = zeros(40);
c = [a b; b a];
d = [b b; a a];
A = 10 * (c + d);
M = c .* d;
D = c ./ d;
S = c - d;
subplot(2, 3, 1); imshow(c); title('C');
subplot(2, 3, 2); imshow(d); title('D');
subplot(2, 3, 3); imshow(A); title('A');
subplot(2, 3, 4); imshow(M); title('M');
subplot(2, 3, 5); imshow(D); title('D');
subplot(2, 3, 6); imshow(S); title('S');

a = imread('baby.jpg');
% Obtain the bit planes
b0 = double(bitget(a, 1));
b1 = double(bitget(a, 2));
b2 = double(bitget(a, 3));
b3 = double(bitget(a, 4));
b4 = double(bitget(a, 5));
b5 = double(bitget(a, 6));
b6 = double(bitget(a, 7));
b7 = double(bitget(a, 8));
% Show the images
subplot(3, 3, 1); imshow(a); title('Original');
subplot(3, 3, 2); imshow(b0); title('b0');
subplot(3, 3, 3); imshow(b1); title('b1');
subplot(3, 3, 4); imshow(b2); title('b2');
subplot(3, 3, 5); imshow(b3); title('b3');
subplot(3, 3, 6); imshow(b4); title('b4');
subplot(3, 3, 7); imshow(b5); title('b5');
subplot(3, 3, 8); imshow(b6); title('b6');
subplot(3, 3, 9); imshow(b7); title('b7');

a = imread('cameraman.tif');
b = imhist(a);
c = histeq(a);
d = imhist(c);
subplot(2, 2, 1); imshow(a); title('Original');
subplot(2, 2, 2); imhist(a); title('Histogram');
subplot(2, 2, 3); imshow(c); title('Equalization');
subplot(2, 2, 4); imhist(c); title('Hist-Equal');
