%% MUST DOWNLOAD 'text8' and unzip into same directory
% https://cs.fit.edu/~mmahoney/compression/text8.zip

corpus_file = 'text8'
MIN_WORD_COUNT = 50

%% load corpus
fid = fopen(corpus_file);
corpus = textscan(fid, '%s');
fclose(fid);

%% index unique words in corpus
[words,~,idx] = unique(corpus{1});
N = length(idx);
M = length(words);
counts = accumarray(idx,1);

%% sort words by count
[~, sidx] = sort( counts, 'descend' ) ;
swords = words(sidx) ;
scounts = counts(sidx) ;
m = find(scounts<MIN_WORD_COUNT,1);
V = ['UNK';swords(1:m-1)];
H = [sum(scounts(m:end));scounts(1:m-1)];
fprintf('%d unique words\n',m)

%% remove unused word dimensions
q1 = sidx(1:m-1);
q2 = sparse((1:m-1)');
Q = sparse(N,1);
Q(q1) = q2;
T = full(Q(idx))+1;

%% clear space
clear corpus idx sidx words swords counts scounts Q q1 q2 M

%% save corpus data
file = [corpus_file,'_',num2str(MIN_WORD_COUNT),'.mat']
save(file,'T','V','H','N')