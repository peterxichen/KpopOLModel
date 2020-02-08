%% create true theta
truetheta = [5 3 3 4; 6 5 2 1; 8 5 4 2; 4 4 4 3];
%% create permutation
perm = zeros(1000,4);
perm(:,1) = ones(1000,1);
for i = 1:1000
    perm(i,2) = ceil(i/100);
    perm(i,3) = ceil(rem(i,100)/10);
    perm(i,4) = ceil(rem(i,10));
end
for i = 1:1000
    for j = 1:4
        if perm(i,j) == 0
            perm(i,j) = 10;
        end
    end
end
%% create variance of observation
var_obs = [10, 15, 5, 8];
%% matrix of attributes
att = ones(3,4,4);
att(:,:,1) = [1 4 6 5; 1 8 2 5; 1 2 8 3];
att(:,:,2) = [1 5 1 2;1 6 0 2; 1 7 2 1];
att(:,:,3) = [1 2 3 2; 1 3 4 3; 1 1 8 1];
att(:,:,4) = [1 6 5 1; 1 3 7 1; 1 2 8 1];

%% simulation
N = 100;
J = 200;
REWARD_explt = zeros(1,J*N);
% we try N priors
for n = 1:N
    %% create prior
    belieftheta = zeros(4,4);
    vartheta = zeros(4,4,4);
    varlt = zeros(4,4);
    for c = 1:4
        x_training = perm(randperm(1000,250),:);
        y_training = x_training * transpose(truetheta(c,:));
        stddev = std(y_training);
        % we use independent belief lookup table since there is no
        % correlation between the maximum of one category and the maximum
        % of the other
        varlt(c,c) = stddev^2;
        y_training = y_training + normrnd(0,stddev,250,1);
        belieftheta(:,c) = inv(transpose(x_training)*x_training)*transpose(x_training)*y_training;
        vartheta(:,:,c) = inv(transpose(x_training)*x_training)*var_obs(c);
    end
    % for each prior we try J rounds of experiments
    for j = 1:J
        reward_j = myPE_PE(truetheta, belieftheta, att, var_obs, vartheta, varlt);
        %reward_j = myPE_IE(truetheta, belieftheta, att, var_obs, vartheta, varlt, Z_iepar);
        %reward_j = myPE_KG(truetheta, belieftheta, att, var_obs, vartheta, varlt);
        %reward_j = myIE_IE(truetheta, belieftheta, att, var_obs, vartheta, varlt, Z_ielt, Z_iepar);
        %reward_j = myIE_PE(truetheta, belieftheta, att, var_obs, vartheta, varlt, Z_ielt);
        %reward_j = myIE_KG(truetheta, belieftheta, att, var_obs, vartheta, varlt, Z_ielt);
        %reward_j = myKG_KG(truetheta, belieftheta, att, var_obs, vartheta, varlt);
        %reward_j = myKG_IE(truetheta, belieftheta, att, var_obs, vartheta, varlt, Z_iepar);
        %reward_j = myKG_PE(truetheta, belieftheta, att, var_obs, vartheta, varlt);
        REWARD_explt(J*(n-1)+j) = reward_j;
    end
end
%% compute performance
fprintf('Mean = %.4f and standard deviation = %.4f \n', mean(REWARD_explt), std(REWARD_explt))
%% plot a histogram
histogram(REWARD_explt)
title('PE PE')
xlabel('Cumulative Reward')
ylabel('Frequency')
    
