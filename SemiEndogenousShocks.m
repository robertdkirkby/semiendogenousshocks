% Semi-Endogenous Shocks
% The evolution of the (semi-)exogenous shock process z depends on the endogenous state k
% Solve an infinite horizon value function problem for this.
%
% Here I solve the Stochastic Neoclassical Growth model but with the
% variance of the productivity shocks z depending on the capital k.
%
% I then compute the stationary distribution (I am assuming it exists, I am
% not aware of theory covering semi-endogenous shocks; computation seems
% fine). I also simulate some panel data.

%% Set grid sizes
% n_a=2^10;
% n_z=121;  %Note, for the figures to correctly use z=0 this must be an odd number (make it 39)

% Following keep things nice and fast (most of the run time is actually creating pi_z_semiendog which could easily be made much faster)
n_a=2^9;
n_z=21;  %Note, for the figures to correctly use z=0 this must be an odd number (make it 39)

%% Set Parameters
%Discounting rate
Params.beta = 0.9896;

%Parameter values
Params.gamma=2; % CES parameter in utility of consumption 
Params.alpha = 0.4; % share of capital in production fn
Params.delta = 0.0196; % depreciation rate
Params.rho = 0.95; % autocorrelation of exog shock
% Params.sigmasq_epsilon=0.007^2; % std dev of exog shock
Params.sigmasq_epsilon=0.07^2; % std dev of exog shock

%% Compute the steady state
K_ss=((Params.alpha*Params.beta)/(1-Params.beta*(1-Params.delta)))^(1/(1-Params.alpha));
X_ss= Params.delta*K_ss;
%These are not really needed; we just use them to determine the grid on
%capital. I mainly calculate them to stay true to original article.

%% Create grids (grids are defined as a column vectors)

a_grid=20*K_ss*linspace(0,1,n_a).^3'; % Grids should always be declared as column vectors

%% Create the semi-endogenous shock process
% First, use tauchen method to create the grid for z (which is the same for all capital levels; only transition depends on capital)
Params.Tauchen_q=3; %Parameter for the Tauchen method
Params.maxlogk=log(a_grid(end)); % I scale sigmasq_epsilon so that it relates to the maximum value of capital grid
% Remark: This may not be the most appropriate scaling point, you can figure out something more intelligent.

% Note that z_grid is independent of k, it is only pi_z that depends on k
[z_grid, pi_z]=TauchenMethod(0,Params.sigmasq_epsilon,Params.rho,n_z,Params.Tauchen_q, tauchenoptions); %[states, transmatrix]=TauchenMethod_Param(mew,sigmasq,rho,znum,q), transmatix is (z,zprime)
% This pi_z is not actually used for anything (except that
% ValueFnIter_Case1 checks to makes sure it is appropriate size, but is
% then ignored because we use vfoptions.SemiEndogShockFn)

vfoptions.SemiEndogShockFnParamNames={'maxlogk','rho','sigmasq_epsilon','n_z','Tauchen_q',};
vfoptions.SemiEndogShockFn=@(k_val,maxlogk,rho,sigmasq_epsilon,n_z,Tauchen_q) SemiEndogShockFn(k_val,maxlogk,rho,sigmasq_epsilon,n_z,Tauchen_q);
% Note that to pass n_z as a input it has to be stored in Params (as that is where codes look)
Params.n_z=n_z;

% If you want to take a look at what the whole 'semi-endogenous transition
% matrix' looks like (it is created automatically by codes) it will look
% like
pi_z_semiendog=zeros(n_a,n_z,n_z); % Note that endogenous state is the first, then the conditional transition matrix for shocks
parfor ii=1:n_a
    ii % I just do this so can see the progress. Creating pi_z_semiendog is slowest part of this code
    [temp_z_grid,temp_pi_z]=SemiEndogShockFn(a_grid(ii),Params.maxlogk,Params.rho,Params.sigmasq_epsilon,Params.n_z,Params.Tauchen_q);
    pi_z_semiendog(ii,:,:)=temp_pi_z;
    % Note that temp_z_grid is just the same things for all k, and same as
    % z_grid created about 10 lines above, so I don't bother keeping it.
    % I only create it so you can double-check it is same as z_grid
end

% Note, you can actually 'hardcode' by instead setting 
vfoptions.SemiEndogShockFn=pi_z_semiendog;
% VFI Toolkit is automatically creating pi_z_semiendog from
% SemiEndogShockFn using SemiEndogShockFnParamNames, if it detects that it
% is a matrix it will just skip the creating pi_z_semiendog step.

%% Now, create the return function
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime_val, a_val, s_val, gamma, alpha, delta) StochasticNeoClassicalGrowthModel_ReturnFn(aprime_val, a_val, s_val, gamma, alpha, delta);
ReturnFnParamNames={'gamma', 'alpha', 'delta'}; %It is important that these are in same order as they appear in 'StochasticNeoClassicalGrowthModel_ReturnFn'

%% Solve
%Do the value function iteration. Returns both the value function itself,
%and the optimal policy function.
n_d=0; %no d variable
d_grid=0; %no d variable

tic;
% Note that because we are using vfoptions.SemiEndogShockFn the input here for pi_z is not used for anything
[V, Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
time(1)=toc;

%% Simulate agent stationary distribution
simoptions.SemiEndogShockFnParamNames=vfoptions.SemiEndogShockFnParamNames; % Note that this is not acutally going to be used since we are using the 'hardcode' version of SemiEndogShockFn (as this is faster)
simoptions.SemiEndogShockFn=vfoptions.SemiEndogShockFn;

tic;
StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);
time(2)=toc;

%% Simulate panel data
% Note that this will use the same simoptions
FnsToEvaluateParamNames(1).Names={};
FnsToEvaluateFn_k = @(aprime_val,a_val,z_val) a_val; %We just want the aggregate assets (which is this periods state)
FnsToEvaluateParamNames(2).Names={};
FnsToEvaluateFn_z = @(aprime_val,a_val,z_val) z_val; %We just want the aggregate assets (which is this periods state)
FnsToEvaluate={FnsToEvaluateFn_k,FnsToEvaluateFn_z};

% To simulate panel data you need initial states from which to begin the simulations
InitialDist=zeros(n_a,n_z);
InitialDist(150,floor((n_z+1)/2))=1; % moderate level of assets, median shock
% Note: You probably want to drop the first (say 50) time periods as burnin. 
% (the default burnin is zero for panel simulations as you typically want to specify an InitialDist)

% The defaults are
% simoptions.simperiods=50;
% simoptions.numbersims=10^3;

% Note that the following does require the actual z_grid to be correct
tic;
SimPanelValues=SimPanelValues_Case1(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Params,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions);
time(3)=toc;
% SimPanelValues: Number of FnsToEvaluate -by- simperiods -by- numbersims


%%
fprintf('The run times were %d seconds to solve value (and policy) function, %d seconds for stationary distribution, and %d seconds to simulate panel data \n')

figure(1)
surf(V)
title('Value function')

figure(2)
plot(cumsum(sum(StationaryDist,2)))
title('CDF of agents dist over assets')

figure(3)
subplot(2,2,1); 
yyaxis left 
plot(SimPanelValues(1,:,1))
yyaxis right 
plot(SimPanelValues(2,:,1))
subplot(2,2,2); 
yyaxis left 
plot(SimPanelValues(1,:,2))
yyaxis right 
plot(SimPanelValues(2,:,2))
subplot(2,2,3); 
yyaxis left 
plot(SimPanelValues(1,:,3))
yyaxis right 
plot(SimPanelValues(2,:,3))
subplot(2,2,4); 
yyaxis left 
plot(SimPanelValues(1,:,4))
yyaxis right 
plot(SimPanelValues(2,:,4))
title('Simulated time-series for four different agents for k and z')
legend('k','z')



