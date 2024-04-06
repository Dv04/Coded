a = imread('circuit.tif');
e1 = edge(a, 'prewitt');
e2 = edge(a, 'roberts');
e3 = edge(a, 'canny');
e4 = edge(a, 'sobel');

subplot(2, 2, 1); imshow(e1); title('Prewitt');
subplot(2, 2, 2); imshow(e2); title('Roberts');
subplot(2, 2, 3); imshow(e3); title('Canny');
subplot(2, 2, 4); imshow(e4); title('Sobel');

a = imread('cameraman.tif');
gxp = [-1 0 1; -1 0 1; -1 0 1];
gyp = [-1 -1 -1; 0 0 0; 1 1 1];
igxp = conv2(gxp, a);
igyp = conv2(gyp, a);
igxp = igxp / 255;
igyp = igyp / 255;
gp = sqrt(igxp .^ 2 + igyp .^ 2);
e = edge(a, 'prewitt');

subplot(3, 1, 1); imshow(a); title('Original');
subplot(3, 2, 3); imshow(igxp); title('Gx');
subplot(3, 2, 4); imshow(igyp); title('Gy');
subplot(3, 2, 5); imshow(e); title('Prewitt');
subplot(3, 2, 6); imshow(gp); title('G');

a = imread('cameraman.tif');
gxr = [1 0; 0 -1];
gyr = [0 1; -1 0];
igxr = conv2(gxr, a);
igyr = conv2(gyr, a);
gr = sqrt(igxr .^ 2 + igyr .^ 2);
e = edge(a, 'roberts');

subplot(3, 2, 1); imshow(a); title('Original');
subplot(3, 2, 3); imshow(igxr); title('Gx');
subplot(3, 2, 4); imshow(igyr); title('Gy');
subplot(3, 2, 5); imshow(e); title('Roberts');
subplot(3, 2, 6); imshow(gr); title('G');

a = imread('cameraman.tif');
gxc = [-1 0 1; -2 0 2; -1 0 1];
gyc = [-1 -2 -1; 0 0 0; 1 2 1];
igxc = conv2(gxc, a);
igyc = conv2(gyc, a);
igxc = igxc / 255;
igyc = igyc / 255;
gc = sqrt(igxc .^ 2 + igyc .^ 2);
e = edge(a, 'canny');

subplot(3, 2, 1); imshow(a); title('Original');
subplot(3, 2, 3); imshow(igxc); title('Gx');
subplot(3, 2, 4); imshow(igyc); title('Gy');
subplot(3, 2, 5); imshow(e); title('Canny');
subplot(3, 2, 6); imshow(gc); title('G');

a = imread('cameraman.tif');
gxs = [-1 0 1; -2 0 2; -1 0 1];
gys = [1 2 1; 0 0 0; -1 -2 -1];
igxs = conv2(gxs, a);
igys = conv2(gys, a);
igxs = igxs / 255;
igys = igys / 255;
gs = sqrt(igxs .^ 2 + igys .^ 2);
e = edge(a, 'sobel');

subplot(3, 2, 1); imshow(a); title('Original');
subplot(3, 2, 3); imshow(igxs); title('Gx');
subplot(3, 2, 4); imshow(igys); title('Gy');
subplot(3, 2, 5); imshow(e); title('Sobel');
subplot(3, 2, 6); imshow(gs); title('G');
