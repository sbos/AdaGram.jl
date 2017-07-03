  # AdaGram

Adaptive Skip-gram (AdaGram) model is a nonparametric extension of famous Skip-gram model implemented in word2vec software which  is able to learn multiple representations per word capturing different word meanings. This projects implements AdaGram in Julia language.

## Installation

AdaGram is not in the julia package repository yet, so it should be installed in the following way:
```
Pkg.clone("https://github.com/sbos/AdaGram.jl.git")
Pkg.build("AdaGram")
```

## Training a model

The most straightforward way to train a model is to use `train.sh` script. If you run it with no parameters passed or with `--help` option, it will print usage information:
```
usage: train.jl [--window WINDOW] [--workers WORKERS]
                [--min-freq MIN-FREQ] [--remove-top-k REMOVE-TOP-K]
                [--dim DIM] [--prototypes PROTOTYPES] [--alpha ALPHA]
                [--d D] [--subsample SUBSAMPLE] [--context-cut]
                [--epochs EPOCHS] [--init-count INIT-COUNT]
                [--stopwords STOPWORDS]
                [--sense-treshold SENSE-TRESHOLD] [--regex REGEX] [-h] 
                train dict output
```
Here is the description of all parameters:
* `WINDOW` is a half-context size. Useful values are 3-10.
* `WORKERS` is how much parallel processes will be used for training.
* `MIN-FREQ` specifies the minimum word frequency below which a word will be ignored. Useful values are 5-50 depending on the corpora.
* `REMOVE-TOP-K` allows to ignore K most frequent words as well. 
* `DIM` is the dimensionality of learned representations
* `PROTOTYPES` sets the maximum number of learned prototypes. This is the truncating level used in truncated stick-breaking, so the actual amount of memory used depends on this number linearly.
* `ALPHA` is the parameter of underlying Dirichlet process. Larger values of `ALPHA` lead to more meanings discovered. Useful values are 0.05-0.2.
* `D` is used together with `ALPHA` in Pitman-Yor process and `D`=0 turns it into Dirichlet process. We couldn’t get reasonable results with PY, but left the option to change `D`.
* `SUBSAMPLE` is a threshold for subsampling frequent words, similarly to how this is done in word2vec. 
* `—context-cut` option allows to randomly decrease `WINDOW` during the training, which increases training speed with almost no effects on model’s performance
* `EPOCHS` specifies the number of passes over training text, usually one epoch is enough, larger number of epochs is usually required on small corpora.
* `INIT-COUNT` is used for initialization of variational stick-breaking distribution. All prototypes are assigned with zero occurrences except first one which is assigned with `INIT-COUNT`. Zero value means that first prototype gets all occurrences.
* `STOPWORDS` is a path to newline-separated file with list of words that must be ignored during the training
* `SENSE-THRESHOLD` allows to sparse gradients and speed-up training. If the posterior probability of a prototype is blow that threshold then it won’t contribute to parameters’ gradients.
* `REGEX` will be used to filter out words not matching with from the `DICTIONARY` provided
* `train` — path to training text (see Format section below)
* `dict` — path to dictionary file (see Format section below)
* `output` — path for saving trained model.

## Input format

Training text should be formatted as for word2vec. Words are case-sensitive and are assumed to be separated by space characters. All punctuation should be removed unless specially intented to be preserved. You may use `utils/tokenize.sh INPUT_FILE OUTPUT_FILE` for simple tokenization with UNIX utils.

In order to train a model you should also provide a dictionary file with word frequency statistics in the following format:
```
word1   34
word2   456
...
wordN   83
```
AdaGram will assume that provided word frequencies are actually obtained from training file. You may build a dictionary file using `utils/dictionary.sh INPUT_FILE DICT_FILE`.

## Playing with a model

 After model is trained, you may use learned word vectors in the same way as ones learned by word2vec. However, since AdaGram learns several vectors for each word, you may need to _disambiguate_ a word using its context first, in order to determine which vector should be used.

First, load the model and the dictionary:
```
julia> using AdaGram
julia> vm, dict = load_model("PATH_TO_THE_MODEL");
```

To examine how many prototypes were learned for a word, use `expected_pi` function:
```
julia> expected_pi(vm, dict.word2id["apple"])
30-element Array{Float64,1}:
 0.341832   
 0.658164   
 3.13843e-6 
 2.84892e-7 
 2.58649e-8 
 2.34823e-9 
 2.13192e-10
 1.93554e-11
 1.75725e-12
 ⋮          
```
This function returns a `--prototypes`-sized array with prior probability of each prototype. As one may see, in this example only first two prototypes have probabilities significantly larger than zero, and thus we may conclude that only two meanings of word "apple" were discovered. 
We may examine each prototype by looking at its 10 nearest neighbours:
```
julia> nearest_neighbors(vm, dict, "apple", 1, 10)
10-element Array{(Any,Any,Any),1}:
 ("almond",1,0.70396507f0)    
 ("cherry",2,0.69193166f0)    
 ("plum",1,0.690269f0)        
 ("apricot",1,0.6882005f0)    
 ("orange",4,0.6739181f0)     
 ("pecan",1,0.6662803f0)      
 ("pomegranate",1,0.6580653f0)
 ("blueberry",1,0.6509351f0)  
 ("pear",1,0.6484747f0)       
 ("peach",1,0.6313036f0)   
julia> nearest_neighbors(vm, dict, "apple", 2, 10)
10-element Array{(Any,Any,Any),1}:
 ("macintosh",1,0.79053026f0)     
 ("iifx",1,0.71349466f0)          
 ("iigs",1,0.7030192f0)           
 ("computers",1,0.6952761f0)      
 ("kaypro",1,0.6938647f0)         
 ("ipad",1,0.6914306f0)           
 ("pc",4,0.6801078f0)             
 ("ibm",1,0.66797054f0)           
 ("powerpc-based",1,0.66319686f0) 
 ("ibm-compatible",1,0.66120595f0)
```
Now if we provide a context for word "apple" we may obtain posterior probability of each prototype:
```
julia> disambiguate(vm, dict, "apple", split("new iphone was announced today"))
30-element Array{Float64,1}:
 1.27888e-5
 0.999987  
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 ⋮     
julia> disambiguate(vm, dict, "apple", split("fresh tasty breakfast"))
30-element Array{Float64,1}:
 0.999977  
 2.30527e-5
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 0.0       
 ⋮         
```
As one may see, model correctly estimated probabilities of each sense with quite large confidence. Vector corresponding to second prototype of word "apple" can be obtained from `vm.In[:, 2, dict.word2id["apple"]]` and then used as context-aware features of word "apple".

A k-means clustering algorithm is provided to classify words in a given number of clusters (default 100) using their embeddings. The algorithm is taken from the one included in word2vec. Because a word can have different meanings, they can (and should in many cases) be assigned to different clusters. The algorithm writes word meanings above a given prior probability minimum (default 1e-3) and the cluster they belong to.
```
julia> clustering(vm, dict, "clustering_output_file", 10; min_prob=1e-3)
```

Plase refer to [API documentation](https://github.com/sbos/AdaGram.jl/wiki/API) for more detailed usage info.
## Future work
* Full API documentation
* C and python bindings
* Disambiguation into user-provided sense inventory

## References

1. [Project homepage](http://bayesgroup.ru/adagram)
2. Sergey Bartunov, Dmitry Kondrashkin, Anton Osokin, Dmitry Vetrov. Breaking Sticks and Ambiguities with Adaptive Skip-gram.  [ArXiv preprint](http://arxiv.org/abs/1502.07257), 2015
3. Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
