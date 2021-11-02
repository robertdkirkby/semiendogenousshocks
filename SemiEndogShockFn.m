function [z_grid,pi_z]=SemiEndogShockFn(k,maxlogk,rho,sigmasq_epsilon,n_z,Tauchen_q)
% Returns z_grid and pi_z, the later of which depends on the endogenous state
% The first input is the value of the endogenous state.
% Rest are parameters in order listed in vfoptions.SemiEndogShockFnParamNames

% First, create the standard z_grid that I am using
[z_grid, ~]=TauchenMethod(0,sigmasq_epsilon,rho,n_z,Tauchen_q);

% Now, figure out the transition probabilities that make this semi-endogenous

% I am assuming that the process is
% z=rho*z_lag+(1-log(k)/maxlogk)*sigma_epsilon*upsilon
% where upsilon is iid standard normal N(0,1)

% The transition probabilities between states are calculated by integrating the area under the normal distribution conditional on the current value of the state.
% Note that this is exactly what the Tauchen method does, just that we are using a pre-existing grid
% (Clarifying: Yes the pre-existing grid was created with Tauchen method, but for differrent parameter values which were independent of k)
pi_z=nan(n_z,n_z);
% Following lines implement the transition matrix, they are largely just a copy of some code from the TauchenMethod() command.
sigma=sqrt(sigmasq_epsilon)*(1-log(k)/maxlogk); %stddev of e
for ii=1:length(z_grid)
    pi_z(ii,1)=normcdf(z_grid(1)+(z_grid(2)-z_grid(1))/2-rho*z_grid(ii),0,sigma);
    for jj=2:(length(z_grid)-1)
        pi_z(ii,jj)=normcdf(z_grid(jj)+(z_grid(jj+1)-z_grid(jj))/2-rho*z_grid(ii),0,sigma)-normcdf(z_grid(jj)-(z_grid(jj)-z_grid(jj-1))/2-rho*z_grid(ii),0,sigma);
    end
    pi_z(ii,end)=1-normcdf(z_grid(end)-(z_grid(end)-z_grid(end-1))/2-rho*z_grid(ii),0,sigma);
end
z_grid=gpuArray(z_grid);
pi_z=gpuArray(pi_z);
% Double check: cumsum(pi_z,2) shows all each row adding to one.



end
