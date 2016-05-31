function [noisy_digit] = noise3(digit)
noisy_digit = digit;
[m, n] = size(digit);
noise = randperm(m);
noise = noise(1:3); % selects 3 positions to flip
for i=1:3
    noisy_digit(noise(i)) = -noisy_digit(noise(i));
end