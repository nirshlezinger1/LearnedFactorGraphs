function v_fXhat = v_fSumProduct(m_fPriors, s_nConst, m_fTransition)

% Apply sum product detection from computed priors
%
% Syntax
% -------------------------------------------------------
% v_fXhat = v_fSumProduct(m_fPriors, s_nConst, s_nMemSize)
%
% INPUT:
% -------------------------------------------------------
% m_fPriors - evaluated likelihoods for each state at each time instance
% s_nConst - constellation size (positive integer)
% s_nMemSize - channel memory length
% 
%
% OUTPUT:
% -------------------------------------------------------
% v_fXhat - recovered symbols vector


s_nMemSize = round(log(size(m_fTransition,1)) / log(s_nConst));
s_nDataSize = size(m_fPriors, 1);
s_nStates = s_nConst^s_nMemSize;
v_fShat = zeros(1, s_nDataSize);

% Generate state switch matrix - each state appears exactly Const times
m_fStateSwitch = zeros(s_nStates,s_nConst);
for ii=1:s_nStates
    Idx = floor((ii -1)/s_nConst) + 1;
    for ll=1:s_nConst
        m_fStateSwitch(ii,ll) = (s_nStates/s_nConst)*(ll-1) + Idx;
    end
    
end
 

% Compute forward messages path
m_fForward = zeros(s_nStates, 1+s_nDataSize);
% assume that the initial state is only zero (state 1)
m_fForward(1,1) = 1;
for kk=1:s_nDataSize
   for ii=1:s_nStates 
       for ll=1:s_nConst
           s_nNextState = m_fStateSwitch(ii,ll);
           m_fForward(s_nNextState, kk+1) = m_fForward(s_nNextState, kk+1) + ...
                                            m_fForward(ii,kk)*m_fPriors(kk,s_nNextState)...
                                            *m_fTransition(s_nNextState,ii);   
       end
   end
   % Normalize
    m_fForward(:, kk+1) =  m_fForward(:, kk+1) / sum( m_fForward(:, kk+1));
end

% Compute backward messages path
m_fBackward = zeros(s_nStates, s_nDataSize+1);
% the final state does not pass a message
m_fBackward(:,end) = ones(s_nStates,1)/s_nConst;
for kk=s_nDataSize:-1:1
   for ii=1:s_nStates 
       for ll=1:s_nConst
           s_nNextState = m_fStateSwitch(ii,ll);
           m_fBackward(ii, kk) = m_fBackward(ii, kk) + ...
                                 m_fBackward(s_nNextState,kk+1)*m_fPriors(kk,ii)...
                                            *m_fTransition(s_nNextState,ii);
       end
   end
   % Normalize
    m_fBackward(:, kk) =  m_fBackward(:, kk) / sum( m_fBackward(:, kk));
end 


% Compute MAP
 s_fCurState = 1; % Initial state
for kk=1:s_nDataSize
    v_fProb = zeros(s_nConst,1);
    % Loop over possible symbol values
    for ll=1:s_nConst
        % Sum forward-backward products
        
        s_nNextState = m_fStateSwitch(s_fCurState,ll);
        v_fProb(ll) = v_fProb(ll) + ...
            m_fForward(s_fCurState,kk)*m_fBackward(s_nNextState,kk)*m_fTransition(s_nNextState,s_fCurState)*m_fPriors(kk,s_nNextState);
    end
    
    % Select symbol which maximizes APP
    [~, v_fShat(kk)] = max(v_fProb);
    s_fCurState =  m_fStateSwitch(s_fCurState,v_fShat(kk));
end
% pad first memory-1 symbols as the first symbol (zero)
v_fXhat = ones(1, s_nDataSize);
v_fXhat(s_nMemSize:end) = v_fShat(1:end-s_nMemSize+1);
