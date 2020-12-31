function [m_fYtrain, m_fYtrain2, v_fXtrain, m_fYtest,  m_fYtest2, v_fXtest] = GetTraining(v_nTrainID, v_nTestID)

% Get data from Sleep EDF data set
%
% Syntax
% -------------------------------------------------------
% [m_fYtrain, m_fYtrain2, v_fXtrain, m_fYtest,  m_fYtest2, v_fXtest] = GetTraining(v_nTrainID, v_nTestID)
%
% INPUT:
% -------------------------------------------------------
% v_nTrainID - patients used for training
% v_nTestID - patietns used for test
% 
%
% OUTPUT:
% -------------------------------------------------------
% m_fYtrain - training input (EEG signal 1)
% m_fYtrain2 - training input (EEG signal 2)
% v_fXtrain - training labels
% m_fYtest - test input (EEG signal 1)
% m_fYtest2 - test input (EEG signal 2)
% v_fXtest - test labels

m_fYtrain = [];
m_fYtrain2 = [];
v_fXtrain = [];
% Get training set
for kk=1:length(v_nTrainID)
    patient_id = v_nTrainID(kk);
    filename = sprintf('FeaturesAndLabels/n%02d.mat', patient_id);
    load(filename);
    m_fYtrain = [m_fYtrain ; new_features(:,:,1)];
    m_fYtrain2 = [m_fYtrain2 ; new_features(:,:,2)];
    v_fTrainLabels = labels*(1:5)';
    v_fXtrain = [v_fXtrain ; v_fTrainLabels];
end
m_fYtrain = m_fYtrain';
m_fYtrain2 = m_fYtrain2';
v_fXtrain = v_fXtrain';


m_fYtest = [];
m_fYtest2 = [];
v_fXtest = [];
% Get test set
for kk=1:length(v_nTestID)
    patient_id = v_nTestID(kk);
    filename = sprintf('FeaturesAndLabels/n%02d.mat', patient_id);
    load(filename);
    m_fYtest = [m_fYtest ; new_features(:,:,1)];
    m_fYtest2 = [m_fYtest2 ; new_features(:,:,2)];
    v_fTrainLabels = labels*(1:5)';
    v_fXtest = [v_fXtest ; v_fTrainLabels];
end   
m_fYtest = m_fYtest';
m_fYtest2 = m_fYtest2';
v_fXtest = v_fXtest';
