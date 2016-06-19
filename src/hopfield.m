clc, clear all, close all;
addpath 'export_fig'; % export pdf: https://github.com/altmany/export_fig
rng(7); % setting random seed

% generating data
%%%%%%%%%%%%%%%%%
characters = character_generator();
figure('Color', [1 1 1]);
for i = 1:18
    subplot(3,6,i);
    imagesc(reshape(characters(:,i),5,7)',[0,1]);
    axis off; colormap copper;
end

%export_fig('hopfield_chargen.pdf');

% trick to rescal to -1 1 instead of 0 1 (requirement hopfield)
characters = 2*characters -1;

% Training network
%%%%%%%%%%%%%%%%%%
net = newhop(characters(:,1:5));

% plot original first 5 letters
figure('Color', [1 1 1]);
for i = 1:5
    subplot(3,5,i);
    imagesc(reshape(characters(:,i),5,7)',[0,1]);
    axis off; colormap copper;
end

% plot first 5 letters with noise
noisy_digits = zeros(35,5);
for i=1:5
    noisy_digits(:,i)=noise3(characters(:,i));
end
noisy_digits_plot = (noisy_digits+1)/2;
for i = 1:5
    subplot(3,5,i+5);
    imagesc(reshape(noisy_digits_plot(:,i),5,7)',[0,1]);
    axis off; colormap copper;
end

% reconstitute letters
for i = 1:5
    [Y Pf Af] = sim(net, {1 10}, [], {noisy_digits(:,i)});
    C = Y{1,10};
    recon_digits_plot = (C+1)/2;
    subplot(3,5,i+10);
    imagesc(reshape(recon_digits_plot(:,1),5,7)',[0,1]);
    axis off; colormap copper;
end

% Critical loading of network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_rand = 50;
errors = zeros(36, n_rand);

for randg = 1:n_rand % loop over 10 different seeds
    rng(randg);
    %errors = zeros(36, 1);
    for p = 1:36 % loop over a total number of 36 characters
        net = newhop(characters(:,1:p));
        reconstructed_chars = zeros(35, p);
        noisy_digits = zeros(35,p);
        
        for i=1:p % addition of noise
            noisy_digits(:,i)=noise3(characters(:,i));
        end
    
        for i=1:p % reconstruction
            [Y Pf Af] = sim(net, {1 100}, [], {noisy_digits(:,i)});
            reconstructed_chars(:,i) = Y{1,100};
        end
        errors(p, randg) = sum(sum(reconstructed_chars ~= characters(:,1:p)));
    end
end

% averaging errors over multiple random seeds
frac_errors = zeros(36,1);
for p=1:36
    frac_errors(p) = sum(errors(p,:))/n_rand
    frac_errors(p) = frac_errors(p)/(p*35);
end

figure('Color', [1 1 1]);
plot(1:36, frac_errors, 'b-','linewidth',4);
title('Error rate evolution','FontSize',18,'FontWeight', 'normal');
xlabel('Number of patterns','FontSize',14);
ylabel('Error','FontSize',14);

% theoritical capacity
disp(35/(4*log(35)));

res=0;
for i=1:500
    res = exp(i)/i;
    if(res > 10e+66)
        disp(i);
        break;
    end
    disp(res);
end

% Perfect retrieval
%%%%%%%%%%%%%%%%%%%
dense_char = zeros(35*9, 36);
for i=1:36
    position = 1;
    for j=1:35*9
        dense_char(j,i) = characters(position, i);
        if(~mod(j, 9))
            position = position + 1;
        end
    end
end

net = newhop(dense_char(:,1:36));
noisy_digits = zeros(35*9,36);

for i=1:36 % addition of noise
    noisy_digits(:,i)=noise3(dense_char(:,i));
end
for i=1:36 % reconstruction
    [Y Pf Af] = sim(net, {1 100}, [], {noisy_digits(:,i)});
    reconstructed_chars(:,i) = Y{1,100};
end

errors = sum(sum(reconstructed_chars ~= dense_char));
