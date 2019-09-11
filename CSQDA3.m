function IDX_v=CSQDA3(ztest,xt,yt,Sigma1,clambda1,clambda2)
%0.5, 0.25

[n1,p]=size(xt);
[n2,~]=size(yt);
N=n1+n2;
%% Monotone Transformation
ytt=yt;
for j=1:p
    [Fj,Xj] = ecdf(xt(:,j));  % Empirical CDF
    Xj = Xj(2:end);     % Sample points, ECDF duplicates initial point, delete it
    Fj = Fj(2:end);     % Sample values, ECDF duplicates initial point, delete it
    y_med=interp1(Xj,Fj,yt(:,j),'nearest');
    y_med(yt(:,j)<=min(Xj))=0;
    y_med(yt(:,j)>= max(Xj))=1-4/N^2;
    y_med(y_med<=4/N^2)=4/N^2;
    y_med(y_med>=1-4/N^2)=1-4/N^2;
    ytt(:,j)=y_med;
    ytt(:,j)=norminv(y_med)*sqrt(Sigma1(j,j));
end
%% Parameter Estimation
hatmux_c=zeros(p,1);
hatmuy_c=mean(ytt)';  
hatdelta_c=hatmuy_c-hatmux_c;

kendall_X=corr(xt,'type','Kendall');
kendall_Y=corr(yt,'type','Kendall');
hatcorrX_c=sin(pi/2*kendall_X);
stdX=sqrt(diag(Sigma1));
hatSigmaX_c=diag(stdX)*hatcorrX_c*diag(stdX);
hatcorrY_c=sin(pi/2*kendall_Y);
stdY=sqrt(diag(cov(ytt)));
hatSigmaY_c=diag(stdY)*hatcorrY_c*diag(stdY);
%% Estimation of beta and D 
lambda1 = clambda1;%*max(max(abs((hatSigmaX_c*D*hatSigmaY_c+hatSigmaY_c*D*hatSigmaX_c)/2-hatSigmaX_c+hatSigmaY_c))); %2*sqrt(log(p)/N);
lambda2 = clambda2;%*max(max(abs(hatSigmaY_c*beta-hatdelta_c))); %sqrt(log(p)/N);
hatD = DiffNet2(hatSigmaX_c,hatSigmaY_c,N/2,lambda1); %estimation of differetial precision matrix
hatD(abs(hatD)<1e-7)=0;
hatbeta=DiscVec2(hatSigmaY_c,hatmux_c',hatmuy_c',N/2,lambda2); %estimation of discriminant direction
%%
ztt=ztest;
for j=1:p
    [Fj,Xj] = ecdf(xt(:,j));  % Empirical CDF
    Xj = Xj(2:end);     % Sample points, ECDF duplicates initial point, delete it
    Fj = Fj(2:end);     % Sample values, ECDF duplicates initial point, delete it
    z_med=interp1(Xj,Fj,ztest(:,j),'nearest');
    z_med(ztest(:,j)<=min(Xj))=0;
    z_med(ztest(:,j)>= max(Xj))=1-4/N^2;
    z_med(z_med<=4/N^2)=4/N^2;
    z_med(z_med>=1-4/N^2)=1-4/N^2;
    ztt(:,j)=z_med;
    ztt(:,j)=norminv(z_med)*sqrt(Sigma1(j,j));
end
ztest=ztt;
%%
IDX_v=[];
[N,~]=size(ztest);
for i=1:N
    z=ztest(i,:)';
    IDX_v=[IDX_v; (z-hatmux_c)'*hatD*(z-hatmux_c)-2*hatbeta'*(z-hatmux_c/2-hatmuy_c/2)-log(det(hatD*hatSigmaX_c+eye(p)))];
end

end