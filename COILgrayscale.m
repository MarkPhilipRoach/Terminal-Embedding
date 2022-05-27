%%Coilgrayscale

%Converts COIL-100 images into a grayscaled vectorized matrix and saves
%into csv format

my_folder ='Folderpath\coil-100';

location = 'Folderpath\coil-100\*.png';       %  folder in which your images exists
data = imageDatastore(location);       %  Creates a datastore for all images in your folder

N = 100*72;
Data = zeros(128^2+2,N);
for n = 1:N
    img = read(data);
    Agray = img(:, :, 2);
    Data(:,n) = [ceil(n/72); mod(n-1,72); reshape(Agray,[],1)];
end
writematrix(Data,'Folderpath\coilgrayscale.csv')


