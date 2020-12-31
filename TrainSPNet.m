function net = TrainSPNet(m_fXtrain,v_fYtrain ,s_nConst, layers, NetParams)

% Train StaSPNet 
%
% Syntax
% -------------------------------------------------------
% net = TrainSPNet(m_fXtrain,v_fYtrain ,s_nConst, layers, learnRate)
%
% INPUT:
% -------------------------------------------------------
% m_fXtrain - training symobls corresponding to each channel output (memory x training size matrix)
% v_fYtrain - training channel outputs (vector with training size entries)
% s_nConst - constellation size (positive integer)
% layers - neural network model to train / re-train
% NetParams - network parameters
% 
%
% OUTPUT:
% -------------------------------------------------------
% net - trained neural network model

 
s_nM = size(m_fXtrain,1); 

% Combine each set of inputs as a single unique category
v_fCombineVec = s_nConst.^(0:s_nM-1);

% format training to comply with Matlab's deep learning toolbox settings
v_fXcat = categorical((v_fCombineVec*(m_fXtrain-1))');
v_fYcat = num2cell(v_fYtrain,1)'; 

if (NetParams.learnRate == 0)
    learnRate = 0.001;
else
    learnRate = NetParams.learnRate;
end

% % Network parameters
% maxEpochs = 30; %100;
% DropPeriod = 10; % 20
% miniBatchSize = 60;

options = trainingOptions('adam', ... 
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate', learnRate, ...
      'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',NetParams.DropPeriod, ...20, ...
    'MaxEpochs',NetParams.maxEpochs, ...
    'MiniBatchSize',NetParams.miniBatchSize, ...
    'shuffle', 'every-epoch'...
    ...,'GradientThreshold',1, ...
    ...,'L2Regularization', 0.01, ...
    ...,'Verbose',false ...
    );%,'Plots','training-progress'); % This can be unmasked to display training convergence

% Train netowrk
net = trainNetwork(v_fYcat,v_fXcat,layers,options);