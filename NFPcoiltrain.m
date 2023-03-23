%% Data

%Data generated from COIL dataset

%First run COILgrayscale and save csv into same folder as this script

location = 'Folderpath\coil-100\*.png';       %  Folder in which your images exists
data = imageDatastore(location);       %  Creates a datastore for all images in your folder

%% Algorithm
N = 100*72;
d = 128;
delta = 525;
%Generate pointspread function
pointspread = zeros(d^2,1);
for t = 1:d^2
    pointspread(t) = exp(-2*pi*1i*t^2/(2*delta-1));  
end
%Generate mask
maskorg = zeros(d^2,1);
a = max(4,(delta-1)/2);
for t = 1:delta
    maskorg(t) = (exp((-t+1)/a))/((2*delta-1)^(1/4))*exp(2*pi*1i*t^2/(2*delta-1));
end
zeroadd = ceil(d^2/(2*delta-1))*(2*delta-1)-d^2;
Data = zeros(d^2+2+zeroadd,N);
for n = 1:N
    img = read(data);
    Agray = img(:, :, 2);
    object = reshape(im2double(Agray),[],1);
    Yconv = zeros(d^2,1);
    for k = 0:d^2-1
        Yrow = cconv(pointspread, circshift(maskorg,-k).* object,d^2);
        for l = 1
            Yconv(k+1,l) = abs(Yrow(l))^2;
        end
    end  
    [n]
    Data(:,n) = [ceil(n/72); mod(n-1,72); reshape(Yconv,[],1); ones(zeroadd,1)];
end

%% Write Data
writematrix(Data,'NFPcoiltrain.csv')


