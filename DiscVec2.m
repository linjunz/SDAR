function hatbeta = DiscVec2(hatSigmaY,hatmux,hatmuy,n,lambda)
%%
[~,p]=size(hatSigmaY);
n1=n;
hatdelta=hatmuy'-hatmux';
while cond(hatSigmaY)> 1000 
      hatSigmaY = hatSigmaY + 100*sqrt(log(p)/n1)*diag(ones(1,p)); 
end
a=lambda*ones(p,1);

f=ones(2*p,1);
CoeffM=[hatSigmaY -hatSigmaY;-hatSigmaY hatSigmaY];
Coeffb=[a+hatdelta; a-hatdelta];
uv=linprog(f,CoeffM,Coeffb,[],[],zeros(2*p,1));
hatbeta=uv(1:p)-uv((p+1):(2*p));

%IDX = ( (ztest - ones(size(ztest,1),1)*mean(mu, 2)')*beta <=1e-06 ) + 1; %classification
%error=sum(abs(IDX-label_z))/size(ztest,1);
end