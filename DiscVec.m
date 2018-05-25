function hatbeta = DiscVec(xt,yt,lambda)
%%
[n1,p]=size(xt);
[n2,p]=size(yt);
hatmux=mean(xt);  %estimation of mean
hatmuy=mean(yt);
hatdelta=hatmuy'-hatmux';
hatSigma=cov(yt); %estimation of covariance matrix

a=lambda*ones(p,1);

f=ones(2*p,1);
CoeffM=[hatSigma -hatSigma;-hatSigma hatSigma];
Coeffb=[a+hatdelta; a-hatdelta];
uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*p,1));
hatbeta=uv(1:p)-uv((p+1):(2*p));

%IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta <=1e-06 ) + 1; %classification
%error=sum(abs(IDX-label_z))/size(ztest,1);
end