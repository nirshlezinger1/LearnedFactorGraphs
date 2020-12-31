function net = GetSPNet(v_fXtrain, v_fYtrain ,s_nConst, s_nMemSize, NetParams)

% Generate and train a new SPNet network
%
% Syntax
% -------------------------------------------------------
% net = GetSPNet(v_fXtrain, v_fYtrain ,s_nConst, s_nMemSize)
%
% INPUT:
% -------------------------------------------------------
% v_fXtrain - training labels vector
% v_fYtrain - training observatrions (vector / matrix with training size entries)
% s_nConst - dictionary size (positive integer)
% s_nMemSize - channel memory length 
% NetParams - network parameters structure
%
% OUTPUT:
% -------------------------------------------------------
% net - trained neural network model 

% Reshape input symbols into a matrix representation
m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize);
 

% Generate neural network
[inputSize, ~] = size(v_fYtrain);
numClasses = s_nConst^s_nMemSize;


if (NetParams.DNN == 5)
    numHiddenUnits =1200;
else
    numHiddenUnits =100;
end

% Work around converting an LSTM, which is the supported first layer for seuquence proccessing networks in Matlab, into a perceptron with sigmoid activation
LSTMLayer = lstmLayer(numHiddenUnits,'OutputMode','last'... 
    , 'RecurrentWeightsLearnRateFactor', 0 ...
    , 'RecurrentWeightsL2Factor', 0 ...
    );
LSTMLayer.RecurrentWeights = zeros(4*numHiddenUnits,numHiddenUnits);


if (NetParams.DNN == 5)
% Generate network model - 5 layers
layers = [ ...
    sequenceInputLayer(inputSize)
    LSTMLayer    
    fullyConnectedLayer(floor(numHiddenUnits/2)) 
    reluLayer
    fullyConnectedLayer(floor(numHiddenUnits/4))
    reluLayer
    fullyConnectedLayer(floor(numHiddenUnits/8))
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

else
% Generate network model - 3 layers
    layers = [ ...
    sequenceInputLayer(inputSize)
    LSTMLayer    
    fullyConnectedLayer(floor(numHiddenUnits/2)) 
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
end
    

% Train network with default learning rate
net = TrainSPNet(m_fXtrain,v_fYtrain ,s_nConst, layers, NetParams);
 
