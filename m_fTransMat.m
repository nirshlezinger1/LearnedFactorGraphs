function m_fTransition = m_fTransMat(s_nConst, s_nMemSize, v_fXtrain)

% Generate state transition matrix
%
% Syntax
% -------------------------------------------------------
% m_fTransition = m_fTransMat(s_nConst, s_nMemSize)
%
% INPUT:
% ------------------------------------------------------- 
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% 
%
% OUTPUT:
% -------------------------------------------------------
% m_fTransition - transition probaiblity matrix 
%                   row = current state
%                   column = previous state

s_nStates = s_nConst^s_nMemSize; 

% Generate transition matrix
m_fTransition = zeros(s_nStates,s_nStates);
% Uniform transitions
if isempty(v_fXtrain)
    for ii=1:s_nStates
        Idx = mod(ii -1, s_nConst^(s_nMemSize-1));
        for ll=1:s_nStates
            if (floor((ll-1)/s_nConst) == Idx)
                m_fTransition(ii,ll) = m_fTransition(ii,ll) + 1/s_nConst;
            end
        end
    end

% Learn transition from training
else
    % Reshape input symbols into a matrix representation
    m_fXtrain = m_fMyReshape(v_fXtrain, s_nMemSize);
    % Combine each set of inputs as a single unique category
    v_fCombineVec = s_nConst.^(0:s_nMemSize-1);% 
    % Convert to states ordered sequentially
    v_fS = 1+(v_fCombineVec*(m_fXtrain-1))';
    for ii=1:(length(v_fS)-1)
        m_fTransition(v_fS(ii+1),v_fS(ii)) =  m_fTransition(v_fS(ii+1),v_fS(ii)) +1;
    end
    % Normalize columns to unit
    m_fTransition = m_fTransition./sum(m_fTransition);
end




