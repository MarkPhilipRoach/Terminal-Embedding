%% Data
%Data generated from COIL dataset

%First run COILgrayscale and save csv into same folder as this script

data = load('coilgrayscale.csv'); %Load data into script
labels = data(1,:); %Classify labels
rotations = data(2,:); %Classify rotations
images = data(3:16386,:); %Classify one-dimensional column images

%% Algorithm
R = [2 4 8];
M = 8:1:15;
MaxDistTotal = zeros(size(M,2),size(R,2)); %Dummy variable
MinDistTotal = zeros(size(M,2),size(R,2)); %Dummy variable
MaxDistLTotal = zeros(size(M,2),size(R,2)); %Dummy variable
MinDistLTotal = zeros(size(M,2),size(R,2)); %Dummy variable
runtimeTotal = zeros(size(M,2),size(R,2)); %Dummy variable
NonlinearityTotal = zeros(size(M,2),size(R,2)); %Dummy variable
SuccessRateTotal = zeros(size(M,2),size(R,2)); %Dummy variable


epsilon = 0.1; %Choice of epsilon
%m = 100; %Size of embedded space

Tests = 1000; %Number of tests

MaxDist = zeros(Tests,1); %Dummy variable
MinDist = zeros(Tests,1); %Dummy variable
MaxDistL = zeros(Tests,1); %Dummy variable
MinDistL = zeros(Tests,1); %Dummy variable
BigO = zeros(Tests,1); %Dummy variable
Nonlinearity = zeros(Tests,1); %Dummy variable
runtime = zeros(Tests,1); %Dummy variable
SuccessRate = zeros(Tests,1); %Dummy variable
Ident = zeros(Tests,1); %Dummy variable
Ident2 = zeros(Tests,1); %Dummy variable




ncounter = 0;
for r = R
ncounter = ncounter +1;
set = 1:r:7199; %Denotes the set of images taken for training data

Labels = transpose(labels);
Labels(set,:) = [];
 for i = 1:100
     ind = setdiff(1:72,set((i-1)*(72/r)+1:i*(72/r)));
    Set((i-1)*(Tests/100)+1:i*(Tests/100)) = ind(1:(Tests/100)); %Denotes the set of images not used in training data
end


n = size(set,2); 
X = images(:,set); %Loads corresponding images

%d = size(X,1); %Size of images (number of pixels) 

mcounter = 0;

for m = M %Size of embedded space
mcounter = mcounter + 1;   
d = 16384;
Piall = (1/sqrt(m))*randn(m,Tests*d); %Random gaussian embedding





p = 0; %Start of correct classification counter
p2 = 0;
for s = 1:Tests
 
tic %Start timer
%Pi = (1/sqrt(m))*randn(m,d); %Random gaussian embedding
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
cvx_begin quiet
    variable uprime(m)
    minimize sum_square_abs(uprime) - 2*dot(Pi*(xk-u),uprime)
    subject to
        [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end

uembed = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
uembedL = [Pi*u; 0];
Yembed = diag((repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembed,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
YembedL = diag((repmat(uembedL,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])'*(repmat(uembedL,1,size(X,2)) - [Pi*X; zeros(1,size(X,2))])); %Compute distances between test vector and all training vectors in the embedded space
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
[r m s toc labels(testnumber) Ident(s) (p/s)*(100)] %Rolling output
Nonlinearity(s) = (norm(uembed - [Pi*u; 0],2)/norm([Pi*u; 0],2))*100;
end
SuccessRateTotal(mcounter,ncounter) = (p/Tests)*(100);
runtimeTotal(mcounter,ncounter) = mean(mean(runtime));
MaxDistTotal(mcounter,ncounter) = max(max(MaxDist));
MinDistTotal(mcounter,ncounter) = min(min(MinDist));
MaxDistLTotal(mcounter,ncounter) = max(max(MaxDistL));
MinDistLTotal(mcounter,ncounter) = min(min(MinDistL));
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

M = 8:1:15;
R = [2 4 8];
SuccessRateTotal = ones(size(M,2),size(R,2));


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
legend('n=3600','n=1800','n=900')

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
xlim([min(M) 15])
xticks([M])
% Create plot
plot2_1 = plot(M,runtimeTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes2); hold on;
plot2_2 = plot(M,runtimeTotal(:,2),'-go','LineWidth',1.5,'Parent',axes2); hold on;
plot2_3 = plot(M,runtimeTotal(:,3),'-b*','LineWidth',1.5,'Parent',axes2); hold on;

legend('n=3600','n=1800','n=900')

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
xlim([min(M) 15])
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
xlim([min(M) 15])
xticks([M])
% Create plot
plot4_1_1 = plot(M,NonlinearityTotal(:,1),'-r+','LineWidth',1.5,'Parent',axes4); hold on;
plot4_1_2 = plot(M,NonlinearityTotal(:,2),'-go','LineWidth',1.5,'Parent',axes4); hold on;
plot4_1_3 = plot(M,NonlinearityTotal(:,3),'-b*','LineWidth',1.5,'Parent',axes4); hold on;

legend('n=3600','n=1800','n=900')
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
