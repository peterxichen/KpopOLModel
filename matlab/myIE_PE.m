function [sumreward] = myIE_PE(trueTheta, beliefTheta, Att, varObs, varTheta, varLT, Z_ielt)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

mu1 = Att(:,:,1)*transpose(beliefTheta(1,:));
mu2 = Att(:,:,2)*transpose(beliefTheta(2,:));
mu3 = Att(:,:,3)*transpose(beliefTheta(3,:));
mu4 = Att(:,:,4)*transpose(beliefTheta(4,:));

% construct a lookup table
muLT = [max(mu1), max(mu2), max(mu3), max(mu4)];

muPar = [transpose(mu1); transpose(mu2); transpose(mu3); transpose(mu4)];

% collecting rewards in this vector
reward = zeros(1,28);

% tunable parameters
Z_ielt = 2; % lookup table

for m = 1:28
    %disp(m)
    %% Interval Estimation
    % idx = index of categorty
    [val, idx] = max(muLT+Z_ielt*sqrt([varLT(1,1),varLT(2,2),varLT(3,3),varLT(4,4)]));    
    
    %% Pure Exploitation
    % sub_idx = index of choice in the category that we choose
    [~, sub_idx] = max(muPar(idx,:));
    
    %% observation
    %disp(trueTheta(idx,:))
    %disp(Att(sub_idx,:,idx))
    %disp(sub_idx)
    %disp(idx)
    %disp(size(Att))
    obs = trueTheta(idx,:)*transpose(Att(sub_idx,:,idx)) + randn(1)*sqrt(varObs(idx));
    reward(m) = obs;
    %% Updating equations
    % update the theta
    gamma = varObs(idx) + Att(sub_idx, :, idx)*varTheta(:,:,idx)*transpose(Att(sub_idx, :, idx));
    err = varObs(idx) - beliefTheta(idx,:)*transpose(Att(sub_idx,:,idx));
    %disp(((1/gamma)*(1/varObs(idx))*err)*varTheta(:,:,idx)*transpose(Att(sub_idx,:,idx)))
    beliefTheta(idx,:) = beliefTheta(idx,:) + transpose(((1/gamma)*(1/varObs(idx))*err)*varTheta(:,:,idx)*transpose(Att(sub_idx,:,idx)));
    varTheta = varTheta - (1/gamma)*(varTheta(:,:,idx))*transpose(Att(sub_idx, :, idx))*Att(sub_idx,:,idx)*varTheta(:,:,idx);
    mu1 = Att(:,:,1)*transpose(beliefTheta(1,:));
    mu2 = Att(:,:,2)*transpose(beliefTheta(2,:));
    mu3 = Att(:,:,3)*transpose(beliefTheta(3,:));
    mu4 = Att(:,:,4)*transpose(beliefTheta(4,:));
    muPar = [transpose(mu1); transpose(mu2); transpose(mu3); transpose(mu4)];
    % update the lookup table
    e_x = zeros(4,1);
    e_x(idx) = 1;
    muLT = muLT + ((obs - val)/(varObs(idx) + varLT(idx,idx)))*transpose(varLT*e_x);
    varLT = varLT - (varLT*e_x*transpose(e_x)*varLT)/(varObs(idx)+varLT(idx,idx));
end
sumreward = sum(reward);

