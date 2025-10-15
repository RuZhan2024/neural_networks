% xor - lms demo for the XOR problem
%   Trains on one pattern at a time, to make demo more interesting.

% Calculate and plot the training set.
NPATS = 4;
Patterns = [ 0 0 1 1; 0 1 0 1]
Desired = [0 1 1 0];
LearnRate = 0.03;

PlotPats(Patterns,Desired)

Inputs = [ones(1,NPATS); Patterns];
Weights = [0 -1 -1];

for i = 1:100

  k = 1+rem(i,NPATS);
  Result = (Weights * Inputs(:,k)) > 0;

  if Result == Desired, break, end

  Weights = Weights + LearnRate*(Desired(k)-Result) * Inputs(:,k)';
  fprintf('%2d.  Weights = ',i);
  disp(Weights);

  PlotBoundary(Weights,i,0)
  pause(0.5)

end

PlotBoundary(Weights,i,1)
