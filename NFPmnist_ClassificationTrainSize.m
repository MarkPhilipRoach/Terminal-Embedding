%% Data
%Data generated from mnist dataset, available at
%https://pjreddie.com/projects/mnist-in-csv/. First run NFPmnisttrain

data = load('NFPmnisttrain.csv'); %Load data into script
labels = data(1,:); %Classify labels
images = data(2:size(data,1),:); %Classify one-dimensional column images

%% Algorithm
d = size(images,1);
N = [8000 4000];
M = 10:2:24;
runtimeTotal = zeros(size(M,2),size(N,2)); %Dummy variable
SuccessRateTotal = zeros(size(M,2),2*size(N,2)+1); %Dummy variable

Tests = 1000; %Number of tests


runtime = zeros(Tests,1); %Dummy variable
Ident = zeros(Tests,1); %Dummy variable
IdentL = zeros(Tests,1); %Dummy variable

Set = zeros(1,Tests);  
Set1 = setdiff(1:size(images,2),1:max(N));
 for i = 0:9
    ind = find(labels(Set1)==i);
    ind = ind(randperm(length(ind)));
    Set((Tests/10)*i+1:(Tests/10)*(i+1)) = Set1(ind(1:Tests/10)); %Denotes the set of images not used in training data
 end
Set = Set(randperm(length(Set)));

set1 = 1:max(N);
ncounter = 0;
for n = N
ncounter = ncounter+1; 
set = zeros(1,n);
 
for i = 0:9
    ind = find(labels(set1)==i);
    ind = ind(randperm(length(ind)));
    set((n/10)*i+1:(n/10)*(i+1)) = ind(1:n/10); %Denotes the set of images taken for training data
end
set = set(randperm(length(set)));
set1 = set;

X = images(:,set); %Loads corresponding images

%% Noncompression
if n == max(N)
p = 0;
for s = 1:Tests
tic %Start timer
testnumber = Set(s);
u = images(:,testnumber); %Corresponding image
Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
Ident(s) = labels(set(k)); %Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
[n/1000 s toc labels(testnumber) Ident(s) (p/s)*(100)]
end

SuccessRateTotal(:,5) = (p/Tests)*(100);
else
end

%% Compression

mcounter = 0;
for m = M %Size of embedded space
mcounter = mcounter + 1;   
Piall = (1/sqrt(m))*randn(m,Tests*d); %Random gaussian embedding

p = 0; %Start of correct classification counter
pL = 0;
for s = 1:Tests
epsilon = 0.1; %Choice of epsilon
tic %Start timer
Pi = Piall(:,(s-1)*d+1:s*d);

testnumber = Set(s);
u = images(:,testnumber); %Corresponding image

Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
xk = X(:,k); %Closest point in R^d
Mx = X - repmat(xk,1,size(X,2)); %Distances from training data to closest point
Mu = repmat(u - xk,1,size(X,2)); %Distance from u to closest point
%uprime = zeros(m,1); %Initial point for cvx algorithm
uprime = Pi*u; %Initial point for cvx algorithm
cvx_begin quiet %Begin cvx
    variable uprime(m,1) %Variable
    minimize sum_square_abs(uprime) - 2*dot(Pi*(xk-u),uprime) %Minimizer function
    subject to
        [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end %End cvx
uembed = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
Yembed = diag((repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
[~,k] = min(Yembed); %Compute index of closest point in R^m+1
Ident(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
uembedL = [Pi*u; 0];
YembedL = diag((repmat(uembedL,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembedL,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
[~,k] = min(YembedL); %Compute index of closest point in R^m+1
IdentL(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == IdentL(s)
    pL = pL+1; %Identifies correctness of classification
else
end
runtime(s) = toc; %End timer
[ncounter m s toc labels(testnumber) Ident(s) (p/s)*(100)] %Rolling output
end
SuccessRateTotal(mcounter,ncounter) = (p/Tests)*(100);
SuccessRateTotal(mcounter,ncounter+2) = (pL/Tests)*(100);
runtimeTotal(mcounter,ncounter) = mean(mean(runtime));
end
end

%% Write\Read Matrix

%Save or upload data
% writematrix(SuccessRateTotal,'Folderpath\SuccessRateTotal.csv')
% writematrix(MaxDistTotal,'Folderpath\MaxDistTotal.csv')
% writematrix(MinDistTotal,'Folderpath\MinDistTotal.csv')
% writematrix(MaxDistLTotal,'Folderpath\MaxDistLTotal.csv')
% writematrix(MinDistLTotal,'Folderpath\MinDistLTotal.csv')
% writematrix(NonlinearityTotal,'Folderpath\NonlinearityTotal.csv')
% writematrix(runtimeTotal,'Folderpath\runtimeTotal.csv')

% SuccessRateTotal = load('SuccessRateTotal.csv');
% MaxDistTotal = load('MaxDistTotal.csv');
% MinDistTotal = load('MinDistTotal.csv');
% MaxDistLTotal = load('MaxDistLTotal.csv');
% MinDistLTotal = load('MinDistLTotal.csv');
% NonlinearityTotal = load('NonlinearityTotal.csv');
% runtimeTotal = load('runtimeTotal.csv');

%% Plotting figures
%Figure 1: m vs Successful Classification Perentage
% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

xlim([min(M) max(M)])
ylim([min(min(SuccessRateTotal)) 90])
xticks([M])
% Create plot
plot1_1 = plot(M,SuccessRateTotal(:,5),'--k','LineWidth',2.5,'Parent',axes1); hold on;
plot1_2 = plot(M,SuccessRateTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes1); hold on;
plot1_3 = plot(M,SuccessRateTotal(:,2),'-g*','LineWidth',1.5,'Parent',axes1); hold on;
plot1_4 = plot(M,SuccessRateTotal(:,3),'--r','LineWidth',1.5,'Parent',axes1); hold on;
plot1_5 = plot(M,SuccessRateTotal(:,4),'--g','LineWidth',1.5,'Parent',axes1); hold on;
legend('NearestNeighbor: n=8000','n=8000','n=4000','Linear: n=8000','Linear: n=4000')

% Create ylabel
ylabel({'Successful Classification Perentage'});

% Create xlabel
xlabel({'m'});

% Create title
title({'m vs Classification Success'});

box(axes1,'on');
hold(axes1,'off');

% Create legend
legend(axes1,'show');



