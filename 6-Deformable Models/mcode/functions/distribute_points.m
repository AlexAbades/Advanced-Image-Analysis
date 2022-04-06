function C_new = distribute_points(C)
% DISTRIBUTE_POINTS   Distributes snake points equidistantly
%   Author: vand@dtu.dk

C = C([1:end,1],:); % closing the curve
N = size(C,1); % number of points (+ 1, due to closing)
dist = sqrt(sum(diff(C).^2,2)); % edge segment lengths
t = [0;cumsum(dist)]; % current positions along the snake
tq = linspace(0,t(end),N)'; % equidistant positions
C_new(:,1) = interp1(t,C(:,1),tq); % distributed x
C_new(:,2) = interp1(t,C(:,2),tq); % distributed y
C_new = C_new(1:end-1,:); % opening the curve again
