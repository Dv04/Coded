a = imread ('cameraman.tif');
a = im2double (a);

[m, n] = size (a);

for i = 1:m

    for j = 1:n
        a_1(i, j) = a(i, j) + sin(5 * i) + sin(5 * j);
    end

end

% Fourier transform of image
A = fft2(a_1);
% shifting origin
A_shift = fftshift (A);
% Magnitude of A shift (Freq. domain repre.)
A_real = abs (A_shift);
% Cut - Off frequency OR Standard deviation sigma
D0 = 52;
% Width of rejection
W = 10;

for u = 1:m

    for v = 1:n
        D = sqrt ((u - m / 2) .^ 2 + (v - n / 2) .^ 2);
        H (u, v) = 1 - exp(- (1/2) * ((D ^ 2 - D0 ^ 2) / (D * W)) ^ 2);
    end

end

H_high = H .* A_shift;
H_high_real = H .* A_real;
H_high_shift = ifftshift (H_high);
H_high_image = ifft2(H_high_shift);

subplot (2, 3, 1); imshow(a_1); title('Image with noise');
subplot (2, 3, 2); imshow(uint8 (abs (A))); title('F.T. of i/p without shift');
subplot (2, 3, 3); imshow(uint8 (A_real)); title('Frequency domain image');
subplot (2, 3, 4); imshow(H); title ('Gaussian Band Reject Filter');
subplot (2, 3, 5); mesh (H); title ('Surface plot GBRE')
subplot (2, 3, 6); imshow(abs(H_high_imaqe)); title('Gaussian Band Reject Filtered imaqe');
