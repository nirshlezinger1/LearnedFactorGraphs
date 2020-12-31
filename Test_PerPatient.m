% Sleep pattern analysis
clear all;
close all;
clc;

rng(1);

%% Parameters setting
s_nConst = 5;       % Number of sleep stages
s_nMemSize = 1;     % Model - first order Markov chain 

s_nSetSize = 1;
 

users = [1:13, 15:20];
m_fSER = zeros(2,length(users)); 
TrainSize = 1000;

% Network parameters
NetParams.DNN = 5; % 5 layers
NetParams.maxEpochs = 100;
NetParams.DropPeriod = 20;
NetParams.miniBatchSize = 60;
NetParams.learnRate = 0;


%% Simulation loop
for u=1:length( users)
    kk = users(u);
    % Get Data
    [m_fYtrain, ~, v_fXtrain, m_fYtest,  ~, v_fXtest] = GetTraining(kk, kk);
    % Find index dividing training and test data in patient
    StopIdx = TrainSize -1 + find(v_fXtrain(TrainSize:end) == 1,1); 
     
    
    v_fXtrain = v_fXtrain(1:StopIdx-1);
    m_fYtrain= m_fYtrain(:,1:StopIdx-1);
    v_fXtest = v_fXtest(StopIdx:end);
    m_fYtest = m_fYtest(:,StopIdx:end);

    
    % compute transoition probabilities
    m_fTransition = m_fTransMat(s_nConst, s_nMemSize, v_fXtrain); % + 0.001;
    
    % TODO NIR - normalize inputs 
    m_fYtrainNorm = normc(m_fYtrain); 
    m_fYtestNorm = normc(m_fYtest); 
    
    net = GetSPNet(v_fXtrain, m_fYtrainNorm, s_nConst, s_nMemSize, NetParams);
    % Apply StaSPNet detctor
    [v_fXhat, v_fXhat2] =  ApplySPNet(m_fYtestNorm, net, s_nConst, m_fTransition);
    % Evaluate error rate
    m_fSER(1,u) = mean(v_fXhat ~= v_fXtest);
    m_fSER(2,u) = mean(v_fXhat2 ~= v_fXtest); 

    m_fXhat{u} = v_fXhat;
    m_fXhat2{u} = v_fXhat2;       
    m_fXtest{u} = v_fXtest;
 
    
    toc;
    kk 
end

%% Plot results
% Confusion matrix
v_fX1 = []; v_fX2 = []; v_fXtrue = [];
for ll=1:length(users) v_fX1= [v_fX1, m_fXhat{ll}]; v_fX2= [v_fX2, m_fXhat2{ll}]; v_fXtrue= [v_fXtrue, m_fXtest{ll}]; end
ConfMat = confusionmat(v_fXtrue, v_fX2);
Labels{1} = 'AWA'; Labels{2} = 'REM'; Labels{3} = 'N1'; Labels{4} = 'N2'; Labels{5} = 'N3';

cm2 = confusionchart(ConfMat,Labels,'RowSummary','row-normalized','ColumnSummary','column-normalized', 'XLabel', 'Predicted Sleep State', 'YLabel', 'True Sleep State');

% Accuracy Bar plot
figure;
m_fAcc = 1-m_fSER(1:2,:); 
bar(m_fAcc');



