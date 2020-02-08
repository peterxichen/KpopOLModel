function [sumreward] = myPE_KG(trueTheta, beliefTheta, Att, varObs, varTheta, varLT)
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
    %% Pure Exploitation
    % idx = index of categorty
    [val, idx] = max(muLT);
    %% Knowledge Gradient
    b_kg = Att(:,:,idx)*varTheta(:,:,idx)*transpose(Att(:,:,idx));
    a_kg = beliefTheta(idx,:)*transpose(Att(:,:,idx));
    vecH = zeros(1,4);
    for j = 1:3
        [sorted_b, order] = sort(b_kg(j,:));
        sorted_a = a_kg;
        sorted_a = sorted_a(order);
        c = zeros(1,2);
        for i = 1:2
            c(i) = (sorted_a(i) - sorted_a(i+1))/(sorted_b(i+1)-sorted_b(i));
        end
        if c(1) <= c(2)
            new_c = c;
            new_b = sorted_b;
        else
            new_c = c(1);
            new_b = [sorted_b(1), sorted_b(2)];
        end
            
        M = length(new_c);
        sum_h = 0;
        for i = 1:M
            sum_h = sum_h + (new_b(i+1) - new_b(i))*(-abs(new_c(i))*normcdf(-abs(new_c(i))) + normpdf(-abs(new_c(i))));
        end
        vecH(j) = sum_h;
    end
    [~, sub_idx] = max(vecH);
    %% observation
    obs = trueTheta(idx,:)*transpose(Att(sub_idx,:,idx)) + randn(1)*sqrt(varObs(idx));
    reward(m) = obs;
    %% Updating equations
    % update the theta
    gamma = varObs(idx) + Att(sub_idx, :, idx)*varTheta(:,:,idx)*transpose(Att(sub_idx, :, idx));
    err = varObs(idx) - beliefTheta(idx,:)*transpose(Att(sub_idx,:,idx));
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

