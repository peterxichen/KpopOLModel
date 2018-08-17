function [sumreward] = myIE_IE(trueTheta, beliefTheta, Att, varObs, varTheta, varLT, Z_ielt, Z_iepar)
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

for m = 1:28
    %disp(m)
    %% Interval Estimation
    % idx = index of categorty
    [val, idx] = max(muLT+Z_ielt*sqrt([varLT(1,1),varLT(2,2),varLT(3,3),varLT(4,4)]));
    
    % propogate errors
    varMuPar = [0, 0, 0];
    covar = varTheta(:,:,idx);
    for i = 1:3
        x1 = Att(i,1,idx);
        x2 = Att(i,2,idx);
        x3 = Att(i,3,idx);
        x4 = Att(i,4,idx);
        varMuPar(i) = x1^2*covar(1,1) + x2^2*covar(2,2) + x3^2*covar(3,3) + x4^2*covar(4,4) + 2*x1*x2*covar(1,2) + 2*x1*x3*covar(1,3) + 2*x1*x4*covar(1,4) + 2*x2*x3*covar(2,3) + 2*x2*x4*covar(2,4) + 2*x3*x4*covar(3,4);
    end
    
    
    %% Interval Estimation
    % sub_idx = index of choice in the category that we choose
    [~, sub_idx] = max(muPar(idx,:)+Z_iepar*sqrt(varMuPar));
    
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

