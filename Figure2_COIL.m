%% Data
%Data generated from COIL dataset

%First run COILgrayscale and save csv into same folder as this script

data = load('coilgrayscale.csv'); %Load data into script
labels = data(1,:); %Classify labels
rotations = data(2,:); %Classify rotations
images = data(3:16386,:); %Classify one-dimensional column images

%% Algorithm
R = 2;
M = 8:1:16;
MaxDistTotal = zeros(size(M,2),size(R,2)); %Dummy variable
MinDistTotal = zeros(size(M,2),size(R,2)); %Dummy variable
BigOTotal = zeros(size(M,2),size(R,2)); %Dummy variable
runtimeTotal = zeros(size(M,2),size(R,2)); %Dummy variable
NonlinearityTotal = zeros(size(M,2),size(R,2)); %Dummy variable
SuccessRateTotal = zeros(size(M,2),size(R,2)); %Dummy variable
epsilon = 0.1; %Choice of epsilon
Tests = 1000; %Number of tests


r = R;
rcounter = 1;
set = 1:r:7199; %Denotes the set of images taken for training data
Set = zeros(1,Tests);
Set1 = setdiff(1:7200,set);
Labels = transpose(labels);
Labels(set,:) = [];
 for i = 1:100
    ind = Set1((i-1)*(72/r)+1:i*(72/r));
    ind = ind(randperm(length(ind)));
    Set((i-1)*(Tests/100)+1:i*(Tests/100)) = ind(1:(Tests/100)); %Denotes the set of images used in testing data
 end
 Set = Set(randperm(length(Set)));
 
n = size(set,2); 
X = images(:,set); %Loads corresponding images

d = size(X,1); %Size of images (number of pixels) 
Nonlinearity = zeros(Tests,3); %Dummy variable
SuccessRate = zeros(Tests,1); %Dummy variable
Ident = zeros(Tests,1); %Dummy variable
Ident2 = zeros(Tests,1); %Dummy variable
IdentL = zeros(Tests,1); %Dummy variable

%% Noncompression
p = 0;
for s = 1:Tests
tic %Start timer
testnumber = Set(s);
u = images(:,testnumber); %Corresponding image
Y = sqrt(diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X))); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
Ident(s) = labels(set(k)); %Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
[rcounter s toc (p/s)*(100)]
end
SuccessRateTotal(:, 4) = repmat((p/Tests)*(100),size(M,2),1);

%% Comparison

mcounter = 0;

for m = M %Size of embedded space
mcounter = mcounter + 1;   
Pi = (1/sqrt(m))*randn(m,d); %Random gaussian embedding
p = 0; %Start of correct classification counter
p2 = 0;
pL = 0;
for s = 1:Tests
tic %Start timer
testnumber = Set(s);
u = images(:,testnumber); %Corresponding image
Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
xk = X(:,k); %Closest point in R^d

Mx = X - repmat(xk,1,size(X,2)); %Distances from training data to closest point
Mu = repmat(u - xk,1,size(X,2)); %Distance from u to closest point
uprime = Pi*u; %Initial point for cvx algorithm
cvx_begin quiet
    variable uprime(m,1)
    minimize sum_square_abs(uprime) - 2*(dot(Pi*(xk-u),uprime))
    subject to
        [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end
uembed = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
Yembed = diag((repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
[~,k] = min(Yembed); %Compute index of closest point in R^m+1
Ident(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
uprime = Pi*u; %Initial point for cvx algorithm
cvx_begin quiet
    variable uprime(m,1)
    minimize dot(Pi*(xk-u),uprime)
    subject to
    [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end
uembed2 = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
Yembed2 = diag((repmat(uembed2,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembed2,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
[~,k] = min(Yembed2); %Compute index of closest point in R^m+1
Ident2(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident2(s)
    p2 = p2+1; %Identifies correctness of classification
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
[r m s toc (p/s)*(100) (p2/s)*(100) (pL/s)*(100)] %Rolling output
Nonlinearity(s,1) = (norm(uembed - [Pi*u; 0],2)/norm([Pi*u; 0],2))*100;
Nonlinearity(s,2) = (norm(uembed2 - [Pi*u; 0],2)/norm([Pi*u; 0],2))*100;
Nonlinearity(s,3) = (norm(uembedL - [Pi*u; 0],2)/norm([Pi*u; 0],2))*100;
end
SuccessRateTotal(mcounter, 1) = (p/Tests)*(100);
SuccessRateTotal(mcounter, 2) = (p2/Tests)*(100);
SuccessRateTotal(mcounter, 3) = (pL/Tests)*(100);
NonlinearityTotal(mcounter,1) = mean(mean(Nonlinearity(:,1)));
NonlinearityTotal(mcounter,2) = mean(mean(Nonlinearity(:,2)));
NonlinearityTotal(mcounter,3) = mean(mean(Nonlinearity(:,3)));
end

%% Write\Read Matrix

writematrix(SuccessRateTotal,'Folderpath\SuccessRateTotal.csv')
writematrix(NonlinearityTotal,'Folderpath\NonlinearityTotal.csv')

%SuccessRateTotal = load('Folderpath\SuccessRateTotal.csv');
%NonlinearityTotal = load('Folderpath\NonlinearityTotal.csv');

%% Plotting figures (With Inner Product)
M = 8:1:15;
%Figure 1: m vs Successful Classification Perentage
% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
xlim([min(M) max(M)])
xticks([M])

% Create plot
plot1_1 = plot(M,SuccessRateTotal(1:8,4),'--k','LineWidth',2.5,'Parent',axes1); hold on;
plot1_2 = plot(M,SuccessRateTotal(1:8,1),'-r+','LineWidth',1.5,'Parent',axes1); hold on;
plot1_3 = plot(M,SuccessRateTotal(1:8,2),'-g*','LineWidth',1.5,'Parent',axes1); hold on;
plot1_4 = plot(M,SuccessRateTotal(1:8,3),'-bo','LineWidth',1.5,'Parent',axes1); hold on;
legend('NearestNeighbor','TerminalEmbed','InnerProd','Linear')
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

%Figure 2: m vs nonlinearity

% Create figure
figure2 = figure;

% Create axes
axes2 = axes('Parent',figure2);
hold(axes2,'on');
xlim([min(M) max(M)]) 
xticks([M])
% Create plot
plot2_1 = plot(M,NonlinearityTotal(1:8,1),'-r+','LineWidth',1.5,'Parent',axes2); hold on;
plot2_2 = plot(M,NonlinearityTotal(1:8,2),'-g*','LineWidth',1.5,'Parent',axes2); hold on;
plot2_3 = plot(M,NonlinearityTotal(1:8,3),'-bo','LineWidth',1.5,'Parent',axes2); hold on;

legend('TerminalEmbed','InnerProd','Linear')
% Create ylabel
ylabel({'Nonlinearity'});

% Create xlabel
xlabel({'m'});

% Create title
title({'m vs Nonlinearity '});
box(axes2,'on');
hold(axes2,'off');

% Create legend
legend(axes2,'show');