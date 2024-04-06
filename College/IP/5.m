a = imread('circuit.tif');
r = size(a, 1);
c = size(a, 2);
ah = uint8(zeros(r, c));
nk = r * c;
f = zeros(256, 1);
pdf = zeros(256, 1);
cdf = zeros(256, 1);
cum = zeros(256, 1);
out = zeros(256, 1);

for i = 1:r

    for j = 1:c
        value = a(i, j);
        f(value + 1) = f(value + 1) + 1;
        pdf(value + 1) = f(value + 1) / nk;
    end

end

sum = 0;
L = 255;

for i = 1:size(pdf)
    sum = sum + f(i);
    cum(i) = sum;
    cdf(i) = cum(i) / nk;
    out(i) = round(cdf(i) * L);
end

for i = 1:r

    for j = 1:c
        ah(i, j) = out(a(i, j) + 1);
    end

end

subplot(2, 2, 1); imshow(a); title('original image');
subplot(2, 2, 2); imhist(a); title('Hist');
subplot(2, 2, 3); imshow(ah); title('equalized image');
subplot(2, 2, 4); imhist(ah); title('equalized histogram');

a = imread('circuit.tif');
b = imread('cameraman.tif');

graylevels = 1:256;
a_occurrences = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    a_occurrences(pixel, 1) = sum(a(:) == graylevels(pixel));
end

N = sum(a_occurrences);
a_pdf = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    a_pdf(pixel, 1) = a_occurrences(pixel, 1) / N;
end

a_cdf = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    a_cdf(pixel, 1) = sum(a_pdf(1:pixel));
end

a_sk = a_cdf * 256;
a_newHistVals = round(a_sk, 0);

a_new = zeros(size(a));

for row = 1:size(a, 1)

    for col = 1:size(a, 2)
        a_new(row, col) = a_newHistVals(a(row, col) + 1);
    end

end

a_newOccur = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    a_newOccur(pixel, 1) = sum(a_new(:) == graylevels(pixel));
end

b_occurrences = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    b_occurrences(pixel, 1) = sum(b(:) == graylevels(pixel));
end

N = sum(b_occurrences);
b_pdf = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    b_pdf(pixel, 1) = b_occurrences(pixel, 1) / N;
end

b_cdf = zeros(size(graylevels, 1));

for pixel = 1:length(graylevels)
    b_cdf(pixel, 1) = sum(b_pdf(1:pixel));
end

b_sk = b_cdf * 256;
b_newHistVals = round(b_sk, 0);

b_new = zeros(size(b));

for row = 1:size(b, 1)

    for col = 1:size(b, 2)
        b_new(row, col) = b_newHistVals(b(row, col) + 1);
    end

end

b_newMatched = zeros(size(a));

for row = 1:size(b_new, 1)

    for col = 1:size(b_new, 2)
        intensity = b_new(row, col);
        [dummy, matching_index] = min(abs(a_newHistVals - intensity));
        b_newMatched(row, col) = a_newHistVals(matching_index);
    end

end

a_new = uint8(a_new);
b_new = uint8(b_new);
b_newMatched = uint8(b_newMatched);

subplot(3, 2, 1); imshow(a); title('Original A');
subplot(3, 2, 2); imshow(b); title('Original B');
subplot(3, 2, 3); imhist(a_new); title('Histogram A');
subplot(3, 2, 4); imhist(b_new); title('Histogram B');
subplot(3, 2, 5); imhist(b_newMatched); title('Histogram B Matched');
subplot(3, 2, 6); imshow(b_newMatched); title('Histogram B Matched Image');
