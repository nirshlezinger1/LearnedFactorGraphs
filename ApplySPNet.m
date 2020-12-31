function [v_fXhat, v_fXhat2] = ApplySPNet(v_fY, net, s_nConst, m_fTransition)

% Apply SPNet to observations
%
% Syntax
% -------------------------------------------------------
% [v_fXhat, v_fXhat2] =  ApplySPNet(v_fY, net, GMModel, s_nConst, m_fTransition)
% INPUT:
% -------------------------------------------------------
% v_fY - channel output vector
% net - trained neural network model
% GMModel - trained mixture model PDF estimate
% s_nConst - constellation size (positive integer)
% m_fTransition - state transition matrix
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector using direct application
% v_fXhat2 - recovered symbols vector using StaSPNet 
 
% Use network to compute likelihood function 
m_fLikelihood = predict(net, num2cell(v_fY,1)); 
 
% Apply argmax to softmax output
[~, v_fXhat] = max(m_fLikelihood');
% Apply sum product
v_fXhat2 = v_fSumProduct(m_fLikelihood, s_nConst, m_fTransition);