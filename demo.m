%% load and preprocess text data (text8)
preproc

%corpus_file = 'text8'
%MIN_WORD_COUNT = 50
%file=[corpus_file,'_',num2str(MIN_WORD_COUNT),'.mat']; load(file)

%% parameters
e = 100;        % embedding size
n = length(V);  % vocab size
w = 7;          % window size
neg = 10;        % # neg samples
r = 0.025;      % learning rate
epoch = 100;     % # epochs
m = round(w/2);
t = [ones(1,w-1),zeros(1,neg*(w-1))];
nn = neg*(w-1);

%% initialize input embeddings (uniform)
W = rand(n,e)*2-1;

%% init output embeddings (trunc normal)
s = 1/sqrt(e);
Z = normrnd(0,s,[1.5*n*e,1]);
Z = Z(logical((Z>-s*2) .* (Z<s*2)));
Z = reshape(Z(1:n*e),e,n);

%% sampling architecture
A = fliplr(toeplitz(1:N,ones(1,w)));
A = A(w:end,:);
S = T(A);
N = N-w+1;

%% train
tic
for iter=1:epoch
    fprintf('\nEPOCH %d\n',iter)
    p=randperm(N); i=1;
    
    % sample 1M (random) lines from corpus on each epoch
    p=p(1:1000000);
    
    for j=p
        i=i+1; if ~mod(i,100000),fprintf('\tsampled %d lines\n',i);end
        
        %% forward update (one sample)
        v = S(j,:);
        k = v(m);
        h = W(k,:)';                % input (target/middle) word embedding
        v(m)=[];                    % pos samples (context)
        u = randperm(n,nn);         % rand pick neg samples
        q = [v,u];
        
        %% backprop updates
        d = logsig(h'*Z(:,q))-t;    % delta
        W(k,:)= W(k,:)-r*d*Z(:,q)'; % input (target) layer
        Z(:,q) = Z(:,q)-r*h*d;      % output (context) layer
    end
    
    %% save weights (each epoch)
    file = ['W_d',num2str(e),'w',num2str(w),'n',num2str(neg),'e',num2str(iter),'.mat'];
    save(file,'W');
    fprintf('\tweights saved to %s\n',file)
    toc
end

