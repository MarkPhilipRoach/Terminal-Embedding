%% Data
%Data generated from mnist dataset, available at https://pjreddie.com/projects/mnist-in-csv/

data = load('mnist_train.csv'); %Load data into script
labels = data(:,1); %Classify labels
images = transpose(data(:,2:785)); %Classify one-dimensional column images

%% Algorithm
N = [4000 2000 1000];
M = 12:2:26;
MaxDistTotal = zeros(size(M,2),size(N,2)); %Dummy variable
MinDistTotal = zeros(size(M,2),size(N,2)); %Dummy variable
MaxDistLTotal = zeros(size(M,2),size(N,2)); %Dummy variable
MinDistLTotal = zeros(size(M,2),size(N,2)); %Dummy variable
runtimeTotal = zeros(size(M,2),size(N,2)); %Dummy variable
NonlinearityTotal = zeros(size(M,2),size(N,2)); %Dummy variable
SuccessRateTotal = zeros(size(M,2),size(N,2)); %Dummy variable

Tests = 1000; %Number of tests


MaxDist = zeros(Tests,1); %Dummy variable
MinDist = zeros(Tests,1); %Dummy variable
MaxDistL = zeros(Tests,1); %Dummy variable
MinDistL = zeros(Tests,1); %Dummy variable
Nonlinearity = zeros(Tests,1); %Dummy variable
runtime = zeros(Tests,1); %Dummy variable
Ident = zeros(Tests,1); %Dummy variable

mcounter = 0;

for m = M %Size of embedded space
mcounter = mcounter + 1;   
d = 784;
Piall = (1/sqrt(m))*randn(m,Tests*d); %Random gaussian embedding
ncounter = 0;
for n = N
ncounter = ncounter+1;
for i = 0:9
    ind = find(labels(:,1)==i);
    [i size(ind)];
    ind = ind(randperm(length(ind)));
    set((n/10)*i+1:(n/10)*(i+1)) = ind(1:n/10); %Denotes the set of images taken for training data
end
Labels = labels;
Labels(set,:) = [];
 for i = 0:9
    ind = find(Labels(:,1)==i);
    ind = ind(randperm(length(ind)));
    Set((Tests/10)*i+1:(Tests/10)*(i+1)) = ind(1:Tests/10); %Denotes the set of images not used in training data
 end
X = images(:,set(1:n)); %Loads corresponding images

p = 0; %Start of correct classification counter
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
YembedL = diag((repmat([Pi*u; 0],1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat([Pi*u; 0],1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
Dist = sqrt(Yembed./Y); %Compute distortion
DistL = sqrt(YembedL./Y); %Compute distortion
MaxDist(s) = max(Dist); %Compute maximum distortion
MinDist(s) = min(Dist); %Compute minimum distortion
MaxDistL(s) = max(DistL); %Compute maximum distortion
MinDistL(s) = min(DistL); %Compute minimum distortion

[~,k] = min(Yembed); %Compute index of closest point in R^m+1
Ident(s) = labels(set(k));%Finds the label of the closest point in R^m+1
if labels(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
runtime(s) = toc; %End timer
[ncounter m s toc labels(testnumber) Ident(s) (p/s)*(100)] %Rolling output
Nonlinearity(s) = (norm(uembed - [Pi*u; 0],2)/norm([Pi*u; 0],2))*100;
end
SuccessRateTotal(mcounter,ncounter) = (p/Tests)*(100);
runtimeTotal(mcounter,ncounter) = mean(mean(runtime));
MaxDistTotal(mcounter,1) = max(max(MaxDist));
MinDistTotal(mcounter,1) = min(min(MinDist));
MaxDistLTotal(mcounter,1) = max(max(MaxDistL));
MinDistLTotal(mcounter,1) = min(min(MinDistL));
NonlinearityTotal(mcounter,ncounter) = mean(mean(Nonlinearity));
end
end

%% Write\Read Matrix

%Save or upload data
writematrix(SuccessRateTotal,'Folderpath\SuccessRateTotal.csv')
writematrix(MaxDistTotal,'Folderpath\MaxDistTotal.csv')
writematrix(MinDistTotal,'Folderpath\MinDistTotal.csv')
writematrix(MaxDistLTotal,'Folderpath\MaxDistLTotal.csv')
writematrix(MinDistLTotal,'Folderpath\MinDistLTotal.csv')
writematrix(NonlinearityTotal,'Folderpath\NonlinearityTotal.csv')
writematrix(runtimeTotal,'Folderpath\runtimeTotal.csv')

% SuccessRateTotal = load('SuccessRateTotal.csv');
% MaxDistTotal = load('MaxDistTotal.csv');
% MinDistTotal = load('MinDistTotal.csv');
% MaxDistLTotal = load('MaxDistLTotal.csv');
% MinDistLTotal = load('MinDistLTotal.csv');
% NonlinearityTotal = load('NonlinearityTotal.csv');
% runtimeTotal = load('runtimeTotal.csv');

%% Plotting figures
runtimeTotal = (2/3)*runtimeTotal;
%Figure 1: m vs Successful Classification Perentage
% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

xlim([min(M) max(M)])
xticks([M])
% Create plot
plot1_1 = plot(M,SuccessRateTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes1); hold on;
plot1_2 = plot(M,SuccessRateTotal(:,2),'-go','LineWidth',1.5,'Parent',axes1); hold on;
plot1_3 = plot(M,SuccessRateTotal(:,3),'-b*','LineWidth',1.5,'Parent',axes1); hold on;
legend('n=4000','n=2000','n=1000')

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

%Figure 2: we plot our runtime vs m

% Create figure
figure2 = figure;

% Create axes
axes2 = axes('Parent',figure2);
hold(axes2,'on');
xlim([min(M) max(M)])
xticks([M])
% Create plot
plot2_1 = plot(M,runtimeTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes2); hold on;
plot2_2 = plot(M,runtimeTotal(:,2),'-go','LineWidth',1.5,'Parent',axes2); hold on;
plot2_3 = plot(M,runtimeTotal(:,3),'-b*','LineWidth',1.5,'Parent',axes2); hold on;

legend('n=4000','n=2000','n=1000')

% Create ylabel
ylabel({'Runtime (in seconds)'});

% Create xlabel
xlabel({'m'});

% Create title
title({'m vs Runtime'});

box(axes2,'on');
hold(axes2,'off');

% Create legend
legend(axes2,'show');

%Figure 3: m vs distortion

% Create figure
figure3 = figure;

% Create axes
axes3 = axes('Parent',figure3);
hold(axes3,'on');
xlim([min(M) max(M)])
xticks([M])
% Create plot
%plot3 = plot(10:10:100,[MaxDistTotal MinDistTotal],'LineWidth',1.5,'Parent',axes1);

plot3_1_1 = plot(M,MaxDistTotal(:,1),'-bo','LineWidth',1.5,'Parent',axes3); hold on;
plot3_1_2 = plot(M,MaxDistLTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes3); hold on;
plot3_2_1 = plot(M,MinDistTotal(:,1),'--bo','LineWidth',1.5,'Parent',axes3); hold on;
plot3_2_2 = plot(M,MinDistLTotal(:,1),'--r+','LineWidth',1.5,'Parent',axes3); hold on;

legend('MaxDist: f(u)', 'MaxDist: (\Piu,0)', 'MinDist: (\Piu,0)','MinDist: f(u)')
% Create ylabel
ylabel({'Distortion'});

% Create xlabel
xlabel({'m'});

% Create title
title({'m vs Distortion '});

box(axes3,'on');
hold(axes3,'off');

% Create legend
legend(axes3,'show');

%Figure 4: m vs nonlinearity

% Create figure
figure4 = figure;

% Create axes
axes4 = axes('Parent',figure4);
hold(axes4,'on');
xlim([min(M) max(M)])
xticks([M])
% Create plot
plot4_1_1 = plot(M,NonlinearityTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes4); hold on;
plot4_1_2 = plot(M,NonlinearityTotal(:,2),'-go','LineWidth',1.5,'Parent',axes4); hold on;
plot4_1_3 = plot(M,NonlinearityTotal(:,3),'-b*','LineWidth',1.5,'Parent',axes4); hold on;

legend('n=4000','n=2000','n=1000')
% Create ylabel
ylabel({'Nonlinearity'});

% Create xlabel
xlabel({'m'});

% Create title
title({'m vs Nonlinearity '});

box(axes4,'on');
hold(axes4,'off');

% Create legend
legend(axes4,'show');


