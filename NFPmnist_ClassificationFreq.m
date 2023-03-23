%% Data
%Data generated from mnist dataset, available at
%https://pjreddie.com/projects/mnist-in-csv/. First run NFPmnisttrain

data = load('mnist_train.csv'); %Load data into script
labels = data(:,1); %Classify labels
images = transpose(data(:,2:size(data,2))); %Classify one-dimensional column images


%% Algorithm
N = 8000;

n = N;

F = 1:4:49;

SuccessRateTotal = zeros(size(F,2)); %Dummy variable


Tests = 1000; %Number of tests

runtime = zeros(Tests,1); %Dummy variable
Ident = zeros(Tests,1); %Dummy variable

d = size(images,1);

TrainSize = 8000;
TestSize = 1000;

delta = 25;
pointspread = zeros(d,1);
for t = 1:d
    pointspread(t) = exp(-2*pi*1i*t^2/(2*delta-1)); 
end
set = zeros(1,TrainSize);
Set = zeros(1,TestSize);
 %Compute the mask
maskorg = zeros(d,1);
a = max(4,(delta-1)/2);
for t = 1:delta
    maskorg(t) = (exp((-t+1)/a))/((2*delta-1)^(1/4))*exp(2*pi*1i*t^2/(2*delta-1));
end

for i = 0:9
    ind = find(labels==i);
    ind = ind(randperm(length(ind)));
    set((n/10)*i+1:(n/10)*(i+1)) = ind(1:n/10); %Denotes the set of images taken for training data
end
set = set(randperm(length(set)));
Set1 = setdiff(1:size(images,2),set);
 for i = 0:9
    ind = find(labels(Set1)==i);
    ind = ind(randperm(length(ind)));
    Set((TestSize/10)*i+1:(TestSize/10)*(i+1)) = Set1(ind(1:TestSize/10)); %Denotes the set of images not used in training data
 end
Set = Set(randperm(length(Set)));
Images = images(:,[set(1:TrainSize),Set(1:TestSize)]);
labels2 = transpose([labels(set); labels(Set)]);


XNFP = zeros(d*max(F),TrainSize+TestSize);

for nn = 1:TrainSize+TestSize
object = Images(:,nn);
Yconv = zeros(d,max(F));
for k = 0:d-1
    Yrow = cconv(pointspread, circshift(maskorg,-k).* object,d);
    for l = 1:max(F)
        Yconv(k+1,l) = abs(Yrow(l))^2;
    end
end
[nn]
XNFP(:,nn) = reshape(Yconv,[],1);
end

%% set and Set
Set = zeros(1,Tests);
set = randperm(TrainSize);
Set1 = setdiff(1:TrainSize+TestSize,set);
 for i = 0:9
    ind = find(labels2(Set1)==i);
    ind = ind(randperm(length(ind)));
    Set((Tests/10)*i+1:(Tests/10)*(i+1)) = Set1(ind(1:Tests/10)); %Denotes the set of images not used in training data
 end
Set = Set(randperm(length(Set)));

%% Noncompressed Classification
fcounter = 0;

for Freq = F
fcounter = fcounter+1;


images2 = XNFP(1:d*Freq,:);

X = images2(:,set);

p = 0;
for s = 1:Tests
tic %Start timer
testnumber = Set(s);
u = images2(:,testnumber); %Corresponding image
Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
Ident(s) = labels2(set(k)); %Finds the label of the closest point in R^m+1
if labels2(testnumber) == Ident(s)
    p = p+1; %Identifies correctness of classification
else
end
[Freq s toc labels2(testnumber) Ident(s) (p/s)*(100)]
end
SuccessRateTotal(fcounter) = (p/Tests)*(100);
end

%% Plotting figures

%Figure 1: m vs Successful Classification Perentage
% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

xlim([min(F) max(F)])
ylim([80 100])
xticks([F])
% Create plot
plot1_1 = plot(F,SuccessRateTotal,'-b*','LineWidth',1.5,'Parent',axes1); hold on;


% Create ylabel
ylabel({'Successful Classification Perentage'});

% Create xlabel
xlabel({'Frequency'});

% Create title
title({'Frequency vs Classification Success'});

box(axes1,'on');
hold(axes1,'off');


