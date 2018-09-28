function hatD = DiffNet(xt,yt,lambda1)
%%
[n1,~]=size(xt);
[n2,p]=size(yt);
N=n1+n2;
d=p^2;
hatmux=mean(xt);  %estimation of mean
hatmuy=mean(yt);
mu = [hatmux', hatmuy'];
hatSigmaX=cov(xt); %estimation of covariance matrix
hatSigmaY=cov(yt); %estimation of covariance matrix
%%
M_Delta=hatSigmaX-hatSigmaY;
V_Delta=reshape(M_Delta,[d,1]);

hatSigmaX = hatSigmaX + sqrt(log(p)/N)*diag(ones(1,p));
hatSigmaY = hatSigmaY + sqrt(log(p)/N)*diag(ones(1,p));
fA=@(x) reshape(hatSigmaX*reshape(x,[p,p])*hatSigmaY+hatSigmaY*reshape(x,[p,p])*hatSigmaX,[d,1])/2;
V_D=cg(reshape(inv(hatSigmaY)-inv(hatSigmaX),[d,1]), fA, V_Delta, lambda1);
hatD=reshape(V_D,[p,p]);

end