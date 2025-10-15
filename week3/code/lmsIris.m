% LMS Learning of Irises
load Iris.dat;
Desired = Iris(:,1)';
Patterns = Iris(:,2:5)';
NINPUTS = size(Patterns,1);
NPATS   = size(Patterns,2);
NUNITS  = size(Desired,1);
nrmPat=zeros(NINPUTS,NPATS);
for r=1:NINPUTS, 
	xnrmv=mean(Patterns(r,:)); sigv=std(Patterns(r,:));
    nrmPat(r,:)=(Patterns(r,:)-xnrmv)./sigv;
end
Patterns=nrmPat;
Inputs = [ones(1,NPATS); Patterns];
InitWeights = rand(NUNITS,1+NINPUTS) - 0.5;
Weights = InitWeights;
LearnRate = 0.01;
TSS = Inf;
for i = 1:100,
  NetIn = Weights * Inputs;
  Result = NetIn;
  Error = Result - Desired;
  TSS = sum(sum(Error.^2));
  fprintf('Epoch %.0f:  ',i);
  fprintf('TSS = %6.5f\n',TSS);
  dW = - (Error * Inputs');
  Weights = Weights + LearnRate * dW;
end
