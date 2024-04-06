a = 1;
b = 25;
[row, col] = size(img);
noise = a + (b - a) * randn([row, col]);
gaus = double(img) + noise;
gaus = uint8(gaus);

subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(gaus); title('Gaussian Noise');

a = 1;
b = 25;
row = size(a, 1);
col = size(a, 2);
gau = zeros(row, col);

for i = 1:row

    for j = 1:col
        noise = a + (b - a) * rand();
        gau(i, j) = noise;
    end

end

img_with_noise = double(img) + gau;
img_with_noise = min(max(img_with_noise, 0), 255);
img_with_noise = uint8(img_with_noise);

subplot(2, 2, 1); imshow(img); title('Original');
subplot(2, 2, 2); imshow(img_with_noise); title('Uniform Noise');

a = 0.05;
b = 5;
k = 1 / a;
R = zeros(row, col);

for j = 1:b
    R = R + k * log(1 - rand(row, col));
end

erl = double(img) + R;
erl = uint8(erl);

subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(erl); title('Erlang Noise');

a = 0.02;
b = 0.05;
R = img;
X = rand(row, col);
R(X <= a) = 0;
c = a + b;
d = find(X > a & X <= c);
R(d) = 255;

subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(R); title('Salt and Pepper Noise');

a = 0.05;
k = 1 / a;
R = k * log(1 - rand(row, col));
expo = double(img) + R;
expo = uint8(expo);

subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(expo); title('Exponential Noise');

a = 0;
b = 25;
noise = a + (-b * log(1 - rand([row, col])));
rayel = double(img) + noise;
rayel = uint8(rayel);

subplot(1, 2, 1); imshow(img); title('Original');
subplot(1, 2, 2); imshow(rayel); title('Rayleigh Noise');
