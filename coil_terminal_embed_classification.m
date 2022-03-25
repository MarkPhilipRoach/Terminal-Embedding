%% Data
%Data generated from COIL dataset

data = load('coilgrayscale.csv'); %Load data into script
labels = data(1,:); %Classify labels
rotations = data(2,:); %Classify rotations
images = data(3:16386,:); %Classify one-dimensional column images

%% Algorithm

set = 1:2:7199; %Denotes the set of images taken for training data
X = images(:,set); %Loads corresponding images

d = size(X,1); %Size of images (number of pixels) 
m = 100; %Size of embedded space
epsilon = 0.2; %Choice of epsilon

test = 100; %Number of tests
Ident = zeros(test,1); %Dummy variable
p = 0; %Start of correct classification counter
for s = 1:test
tic %Start timer
Pi = (1/sqrt(m))*randn(m,d); %Random gaussian embedding

testnumber = 72*s; %Rotations considered for test data
u = images(:,testnumber); %Corresponding image
Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[M,k] = min(Y);  %Compute index of closest point in R^d
xk = X(:,k); %Closest point in R^d

Mx = X - repmat(xk,1,size(X,2)); %Distances from training data to closest point
Mu = repmat(u - xk,1,size(X,2)); %Distance from u to closest point
uprime = zeros(m,1); %Initial point for cvx algorithm
cvx_begin quiet
    variable ux(m)
    minimize (norm(uprime - Pi*u,2)) %Minimizing function
    subject to
        norm(uprime,2) <= norm(u - xk,2); %Constraints
        abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1)) <= epsilon*norm(u - xk,2)^2*transpose(diag(Mx'*Mx)); %Constraints
cvx_end

uembed = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
Yembed = diag((repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space

[M,k] = min([Yembed]); %Compute index of closest point in R^m+1
Ident(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
[s toc labels(testnumber) Ident(s) (p/s)*100] %Rolling output
end


