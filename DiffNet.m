function hatD = DiffNet(xt,yt,lambda1)
%%
[n1,p]=size(xt);
[n2,p]=size(yt);
N=n1+n2;
d=p^2;
hatmux=mean(xt);  %estimation of mean
hatmuy=mean(yt);
mu = [hatmux', hatmuy'];
hatSigmaX=cov(xt); %estimation of covariance matrix
hatSigmaY=cov(yt); %estimation of covariance matrix
%%
%M_K=kron(hatSigmaY,hatSigmaX);
M_Delta=hatSigmaX-hatSigmaY;
V_Delta=reshape(M_Delta,[d,1]);

%%
%a=lambda*ones(d,1);
%f=ones(2*d,1);
%CoeffM=[M_K -M_K;-M_K M_K];
%Coeffb=[a+V_Delta; a-V_Delta];
%%
%uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*d,1));
%V_D=uv(1:d)-uv((d+1):(2*d));
%hatD=reshape(V_D,[p,p]);
%%
hatSigmaX = hatSigmaX + sqrt(log(p)/N)*diag(ones(1,p));
hatSigmaY = hatSigmaY + sqrt(log(p)/N)*diag(ones(1,p));
fA=@(x) reshape(hatSigmaX*reshape(x,[p,p])*hatSigmaY+hatSigmaY*reshape(x,[p,p])*hatSigmaX,[d,1])/2;
%betahat2= clime(Sigma_hat\hatdelta, fA, hatdelta, lambda);
V_D=clime(reshape(inv(hatSigmaY)-inv(hatSigmaX),[d,1]), fA, V_Delta, lambda1);
hatD=reshape(V_D,[p,p]);
%%
%Z_t=hatSigmaX*hatD*hatSigmaY/2+hatSigmaY*hatD*hatSigmaX/2-hatSigmaX+hatSigmaY;
%max(max(abs(Z_t)))
%E_t=hatD-D;
%max(max(abs(E_t)))
%%
%IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta <=1e-06 ) + 1; %classification
%error=sum(abs(IDX-label_z))/size(ztest,1);
end