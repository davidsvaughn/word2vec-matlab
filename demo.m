% https://www.tensorflow.org/versions/r0.11/tutorials/word2vec/index.html
% http://norvig.com/ngrams/count_1w.txt
%%

%preproc;

%%
e = 100;        % embedding size
n = length(V);  % vocab size
w = 5;          % window size
neg = 5;        % # neg samples
r = 0.025;      % learning rate
epoch = 20;     % # epochs
m = round(w/2);
t = [ones(1,w-1),zeros(1,neg*(w-1))];
nn = neg*(w-1);

%% initialize
% init input embeddings (uniform)
W = rand(n,e)*2-1;

% init output embeddings (trunc normal)
s = 1/sqrt(e);
Z = normrnd(0,s,[1.5*n*e,1]);
Z = Z(logical((Z>-s*2) .* (Z<s*2)));
Z = reshape(Z(1:n*e),e,n);

% init biases
%B = zeros(n,1);

%% samples
A = fliplr(toeplitz(1:N,ones(1,w)));
A = A(w:end,:);
S = T(A);
N = N-w+1;

%% train
tic
for iter=1:epoch
    %U = sample_table(N,n,nn);
    p=randperm(N);
    i=1;
    for j=p
        i=i+1;
        if ~mod(i,100000),display(i);end
        %% example updates for one sample
        v = S(j,:);
        k = v(m);
        h = W(k,:)';                 % input (target/middle) word embedding
        v(m)=[];                     % pos samples (context)
        %u = U(j,:);                 % rand pick neg samples
        u = randperm(n,nn);
        q = [v,u];
        
        %% backprop updates
        % delta
        d = logsig(h'*Z(:,q))-t;
        % input (target) layer
        W(k,:) = W(k,:)-r*d*Z(:,q)';
        % output (context) layer
        Z(:,q) = Z(:,q)-r*h*d;
    end
    file = ['W_d100w5n5e',num2str(iter),'.mat']
    save(file,'W')
    toc
end

%%
%file = ['W_e100w5n5.mat']
%save(file,'W')
