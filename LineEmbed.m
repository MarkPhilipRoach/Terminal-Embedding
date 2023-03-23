clear all


d = 3;
m = 2;

Tests = 100;
epsilon = 0.1;
uorg1 = zeros(d,Tests);
uorg2 = zeros(d,Tests);
uembed1 = zeros(d,Tests);
uembed2 = zeros(d,Tests);
uembedL = zeros(d,Tests);
Pi = (1/sqrt(m))*randn(m,d);
% N = [rand(1)-0.5; rand(1)-0.5; rand(1)-0.5]; % the normal to the plane
N = [-0.5; -0.1; 0.4]; % the normal to the plane
A = 1000; % size of the rectangle
radius = 0.25*A;
Q = null(N');

for p = 1:3

n = 10^(p+1);
X = Q*((rand(2,n)-0.5)*A);
Xembed = [Pi*X; zeros(1,size(X,2))];

%% Algorithm

for t = 1:Tests
ss = 2*(t-1)/(Tests-1)-1;
u = Q*([ss*(0.5*A);0]);
uorg1(:,t) = u;
Y = diag((repmat(u,1,size(X,2)) - X)'*(repmat(u,1,size(X,2)) - X)); %Compute distances between test vector and all training vectors
[~,k] = min(Y);  %Compute index of closest point in R^d
xk = X(:,k); %Closest point in R^d
Mx = X - repmat(xk,1,size(X,2)); %Distances from training data to closest point
Mu = repmat(u - xk,1,size(X,2)); %Distance from u to closest point
uprime = Pi*u; %Initial point for cvx algorithm
cvx_begin quiet %Begin cvx
    variable uprime(m,1) %Variable
    minimize sum_square_abs(uprime) - 2*dot(Pi*(xk-u),uprime) %Minimizer function
    subject to 
    [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end %End cvx
uembed1(1:m+1,t) = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
uprime = Pi*u; %Initial point for cvx algorithm
cvx_begin quiet %Begin cvx
    variable uprime(m,1) %Variable
    minimize dot(Pi*(xk-u),uprime) %Minimizer function
    subject to 
    [norm(uprime,2) abs(sum(repmat(uprime,1,size(X,2)).*(Pi*Mx),1) - sum(Mu.*Mx,1))] <= [norm(u - xk,2) epsilon*norm(u - xk,2)*diag(Mx'*Mx)']; %Constraints
cvx_end %End cvx
uembed2(:,t) = [Pi*xk + uprime; sqrt(norm(u - xk,2)^2 - norm(uprime,2)^2)]; %Computes the embedded point in R^m+1
[p t ss]
uembedL(:,t) = [Pi*u; zeros(1,size(u,2))];
end


%% Plotting figures
%Figure 1: Original manifold

subplot(4,3,p);

% Create plot
hold on
plot3(X(1,:),X(2,:),X(3,:),'+','MarkerEdgeColor',[0.3010 0.7450 0.9330],'MarkerFaceColor', [0.3010 0.7450 0.9330],'MarkerSize', 0.8);
plot3(uorg1(1,:),uorg1(2,:),uorg1(3,:),'o','MarkerEdgeColor',[0.9290 0.6940 0.1250],'MarkerSize', 2,'MarkerFaceColor', [0.9290 0.6940 0.1250]);
hold off
% Create xlabel
xlabel({'x'});

% Create ylabel
ylabel({'y'});

% Create zlabel
zlabel({'z'});

% Create title
title('Manifold in Original Space', sprintf('n = %d',n));
view(3)

%Figure 2: TerminalEmbed
subplot(4,3,p+3);

% Create plot
hold on
%plot3(Xembed(1,:),Xembed(2,:),Xembed(3,:),'+');
plot3(uembed1(1,:),uembed1(2,:),uembed1(3,:),'o','MarkerEdgeColor','r','MarkerSize', 2,'MarkerFaceColor', 'r');
hold off
% Create xlabel
xlabel({'x'});

% Create ylabel
ylabel({'y'});

% Create zlabel
zlabel({'z'});

% Create title
title({'';'TerminalEmbed'});


%Figure 3: InnerProd
subplot(4,3,p+6);

% Create plot
hold on
%plot3(Xembed(1,:),Xembed(2,:),Xembed(3,:),'+');
plot3(uembed2(1,:),uembed2(2,:),uembed2(3,:),'o','MarkerEdgeColor','g','MarkerSize', 2,'MarkerFaceColor', 'g');
hold off
% Create xlabel
xlabel({'x'});

% Create ylabel
ylabel({'y'});

% Create zlabel
zlabel({'z'});

% Create title
title({'';'InnerProd'});


%Figure 4: Linear
subplot(4,3,p+9);

% Create plot
hold on
%plot3(Xembed(1,:),Xembed(2,:),Xembed(3,:),'+');
plot3(uembedL(1,:),uembedL(2,:),uembedL(3,:),'o','MarkerEdgeColor','b','MarkerSize', 2,'MarkerFaceColor', 'b');
hold off
% Create xlabel
xlabel({'x'});

% Create ylabel
ylabel({'y'});

% Create zlabel
zlabel({'z'});

% Create title
title({'';'Linear'});


end
