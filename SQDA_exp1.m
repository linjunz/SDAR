%%
%Settings%
p = 50;     % Number of variables
N = 400;    % Number of samples
s = 10;     % Number of useful features in the discriminant vector, s<<p
s2= 10;
mu_min = 1; % The magnitude of non-zero entries in beta
pie = 0.5;  % mixing proportion for class 2
%[~, Omega1] = model_er(p,0.05); %Model 1
%[~, Omega1] = model_bern(p,0.5,s); %Model 2
[~, Sigma] = model_exp_decay(p,0.5); %Model 3
Omega1=inv(Sigma)+2*eye(p);

%%
%Generate cov's$
pertb=ones(p,1);
%Omega1=Omega1+10*diag(pertb); % decrease the condition number of Omega1
Sigma1=inv(Omega1);          % Covaiance matrix Sigma1
D=zeros(p,p);                % Generate differetial precision matrix D=Omega2-Omega1
for i=1:s2
    row_i=randsample(1:p,1);
    %row_j=row_i;
    row_j=randsample(1:p,1);
    zz=5/sqrt(s);%normrnd(0,1); 
    D(row_i,row_j)=zz;%5/sqrt(s);
    D(row_j,row_i)=zz;%5/sqrt(s);
    %D(row_i,row_i)=4/sqrt(s);
end
Omega2=D+Omega1;            % Precision matrix Omega2
Sigma2=inv(Omega2);         % Covaiance matrix Sigma2

[~,p_test1] = chol(Sigma1) %Check definite positiveness
[~,p_test2] = chol(Sigma2) %Check definite positiveness

%Generate means&
beta=[mu_min*ones(s,1);zeros(p-s,1)]; % discriminant direction (sparse) 
delta=Sigma2*beta;                    % differential mean delta=mu2-m1
mu1=zeros(p,1);
mu2=delta;

rep=50;
error=[];
error_SLDA=[];
error_oracle=[];
error_LDA=[];
error_QDA=[];

for i_rep=1:rep
i_rep
%% Data generation&
xt = mvnrnd(mu1,Sigma1,N/2);
yt = mvnrnd(mu2,Sigma2,N/2);

%% Parameter estimation
hatmux=mean(xt)';  %estimation of mean
hatmuy=mean(yt)';  %estimation of mean
hatdelta=hatmuy-hatmux;
hatSigmaX=cov(xt); %estimation of covariance matrix
hatSigmaY=cov(yt); %estimation of covariance matrix
hatSigmaX = hatSigmaX + sqrt(log(p)/N)*diag(ones(1,p));
hatSigmaY = hatSigmaY + sqrt(log(p)/N)*diag(ones(1,p));
%regularization&
lambda1 = 0.25*max(max(abs((hatSigmaX*D*hatSigmaY+hatSigmaY*D*hatSigmaX)/2-hatSigmaX+hatSigmaY))); %2*sqrt(log(p)/N);
lambda2 = 0.25*max(max(abs(hatSigmaY*beta-hatdelta))); %sqrt(log(p)/N);
hatD = DiffNet(xt,yt,lambda1); %estimation of differetial precision matrix
%hatD0=inv(hatSigmaY)-inv(hatSigmaX);
%hatD(abs(hatD)>1e-7)=hatD0(abs(hatD)>1e-7);
hatD(abs(hatD)<1e-7)=0;
hatbeta=DiscVec(xt,yt,lambda2); %estimation of discriminant direction
%%
%max(max(abs((hatSigmaX*hatD*hatSigmaY+hatSigmaY*hatD*hatSigmaX)/2-hatSigmaX+hatSigmaY)))
%max(max(abs(hatSigmaY*hatbeta-hatdelta)))
%%
%max(max(abs(D-hatD)))
%max(max(abs(beta-hatbeta)))
%% Testing data
xtest = mvnrnd(mu1,Sigma1,N/2);
ytest = mvnrnd(mu2,Sigma2,N/2);
ztest = [xtest;ytest];

label_z=[ones(N/2,1);ones(N/2,1)+1];

IDX_v=[];
IDX_v_oracle=[];
IDX_v_SLDA=[];
IDX_v_LDA=[];
IDX_v_QDA=[];

%hatD=D;%hatbeta=beta;
for i=1:N
    z=ztest(i,:)';
    IDX_v=[IDX_v; (z-hatmux)'*hatD*(z-hatmux)-2*hatbeta'*(z-hatmux/2-hatmuy/2)-log(det(hatD*hatSigmaX+eye(p)))];
    IDX_v_oracle=[IDX_v_oracle; (z-mu1)'*D*(z-mu1)-2*beta'*(z-mu1/2-mu2/2)-log(det(D*Sigma1+eye(p)))];
    IDX_v_SLDA=[IDX_v_SLDA; -2*hatbeta'*(z-hatmux/2-hatmuy/2)];
    %IDX_v_SLDA=[IDX_v_SLDA; (z-mu1)'*D*(z-mu1)-2*hatbeta'*(z-hatmux/2-hatmuy/2)-log(det(D*Sigma1+eye(p)))];
    IDX_v_LDA=[IDX_v_LDA; -2*(inv(hatSigmaY)\hatdelta)'*(z-hatmux/2-hatmuy/2)];
    IDX_v_QDA=[IDX_v_QDA; (z-hatmux)'*(inv(hatSigmaY)-inv(hatSigmaX))*(z-hatmux)-2*(inv(hatSigmaY)\hatdelta)'*(z-hatmux/2-hatmuy/2)-log(det(hatSigmaX))+log(det(hatSigmaY))];
end

IDX = ( IDX_v <=1e-06 ) + 1; %classification
IDX_SLDA=( IDX_v_SLDA <=1e-06 ) + 1;
IDX_oracle=( IDX_v_oracle <=1e-06 ) + 1;
IDX_LDA=( IDX_v_LDA <=1e-06 ) + 1;
IDX_QDA=( IDX_v_QDA <=1e-06 ) + 1;

%error=sum(abs(IDX-label_z))/size(ztest,1)
%error_SLDA=sum(abs(IDX_SLDA-label_z))/size(ztest,1)
%error_oracle=sum(abs(IDX_oracle-label_z))/size(ztest,1)

error=[error, sum(abs(IDX-label_z))/size(ztest,1)];
error_SLDA=[error_SLDA, sum(abs(IDX_SLDA-label_z))/size(ztest,1)];
error_oracle=[error_oracle, sum(abs(IDX_oracle-label_z))/size(ztest,1)];
error_LDA=[error_LDA, sum(abs(IDX_LDA-label_z))/size(ztest,1)];
error_QDA=[error_QDA, sum(abs(IDX_QDA-label_z))/size(ztest,1)];

end
%%
mean(error_QDA)
std(error_QDA)