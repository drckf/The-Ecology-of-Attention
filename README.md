# The-Ecology-of-Attention
Companion repo to the paper The Ecology of Attention by Charles K. Fisher.

<img src="assets/The_Ecology_of_Attention_Overview.png" 
alt="Schematic comparing species interacting in an ecological community (A) to tokens interacting within a context window (B)." 
width="80%"/>

## Abstract
Attention modules that use key-value associative memories are important components of modern neural networks used for sequential tasks such as language modeling. Recent work has argued that many common key-value memory architectures--including multiple variants of linear attention, state space models, and softmax attention--can be derived from a test-time regression principle. Here, we extend this work in a surprising direction by deriving an exact mapping between associative memories learned through test-time regression and ecological systems described by generalized Lotka-Volterra or replicator dynamics. We show that tokens in a batch behave like species that interact with each other through competition or mutualism, and derive explicit formulas that link the statistics of the data distribution to the carrying capacity and interaction coefficients of each token. In a streaming context, online updating of an associative memory is equivalent to the invasion of an ecosystem by a new species. We use this mapping to derive some novel ecologically inspired attention modules, including a closed-form solution for theoretically optimal gated linear attention.

