%% Data
%Data generated from mnist dataset, available at https://pjreddie.com/projects/mnist-in-csv/

data = load('mnist_train.csv'); %Load data into script
labels = data(:,1); %Classify labels
images = transpose(data(:,2:size(data,2))); %Classify one-dimensional column images

%% Algorithm
d = size(images,1);

TrainSize = 8000;
TestSize = 1000;
Freq = 1;
XNFP = zeros(d*Freq,TrainSize+TestSize);
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
    set((TrainSize/10)*i+1:(TrainSize/10)*(i+1)) = ind(1:TrainSize/10); %Denotes the set of images taken for training data
end

Set1 = setdiff(1:size(images,2),set);
 for i = 0:9
    ind = find(labels(Set1)==i);
    ind = ind(randperm(length(ind)));
    Set((TestSize/10)*i+1:(TestSize/10)*(i+1)) = Set1(ind(1:TestSize/10)); %Denotes the set of images not used in training data
 end

Images = images(:,[set(1:TrainSize),Set(1:TestSize)]);

for nn = 1:TrainSize+TestSize
object = Images(:,nn);
Yconv = zeros(d,Freq);
for k = 0:d-1
    Yrow = cconv(pointspread, circshift(maskorg,-k).* object,d);   
    for l = 1:Freq
        Yconv(k+1,l) = abs(Yrow(l))^2;
    end
end
[nn]
XNFP(:,nn) = reshape(Yconv,[],1);
end


%% Write Data
Data = [transpose([labels(set); labels(Set)]); XNFP];
writematrix(Data,'NFPmnisttrain.csv')
