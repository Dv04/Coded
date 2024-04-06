a = imread('cameraman.tif');
img = im2double(a);
c = 1;
g = 0.6;
[row, col] = size(img);
log_img = zeros(row, col);
power_img = zeros(row, col);

for i = 1:row

    for j = 1:col
        log_img(i, j) = c * log(1 + img(i, j));
        power_img(i, j) = c * img(i, j) .^ g;
    end

end

subplot(1, 3, 1); imshow(a); title('Original');
subplot(1, 3, 2); imshow(log_img); title('Logerithmic');
subplot(1, 3, 3); imshow(power_img); title('Power');

%%
a = imread('cameraman.tif');
[row, col] = size(a);
t2 = 255;
t1 = round(t2 / 1.45);

for i = 1:row

    for j = 1:col

        if a(i, j) > t1 && a(i, j) < t2
            img1(i, j) = a(i, j); % with background
            img2(i, j) = 255; % without background
        else
            img1(i, j) = 0;
            img2(i, j) = 0;
        end

    end

end

subplot(1, 3, 1); imshow(a); title('Original');
subplot(1, 3, 2); imshow(img1); title('With background');
subplot(1, 3, 3); imshow(img2); title('Without background');

%%
i = imread('cameraman.tif');
% k = rgb2gray(i); % if img is RGB
j = imnoise(i, 'salt & pepper', 0.10);
f1 = medfilt2(j);
f2 = medfilt2(j, [3 3]);
f3 = medfilt2(j, [10 10]);

subplot(2, 2, 1); imshow(i); title('Original');
subplot(2, 2, 2); imshow(j); title('Noise');
subplot(2, 2, 3); imshow(f2); title('3x3 mask');
subplot(2, 2, 4); imshow(f3); title('10x10 mask');

%%
i = imread('cameraman.tif');
% k = rgb2gray(i); % if img is RGB
g1 = fspecial('average', [3 3]);
g2 = fspecial('average', [10 10]);

b1 = imfilter(i, g1);
b2 = imfilter(i, g2);

subplot(1, 3, 1); imshow(i); title('Original');
subplot(1, 3, 2); imshow(b1); title('3x3 avg mask');
subplot(1, 3, 3); imshow(b2); title('10x10 avg mask');
