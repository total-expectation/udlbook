# Prince exercises chapter 12 - NLP
## 12.1
```
Problem 12.1 Consider a self-attention mechanism that processes N inputs of length D to
produce N outputs of the same size. How many weights and biases are used to compute the
queries, keys, and values? How many attention weights a[•, •] will there be? How many weights
and biases would there be in a fully connected shallow network relating all DN inputs to all DN
outputs?
```

We have 
* input $D \times N$
* output $D \times N$
* query Q, value V, key K all are $D \times D$

Each of Q, V, K uses one bias matrix of size $D$, that is broadcast to $D \times D$, but only D is learnable, the rest are copied. So in total 3 bias matrices, making it $3D$ parameters. Prince defined each bias initially as a column vector that is broadcast to meet the $D \times D$ requirement.

As for the weights of Q,V,K the matrices are all of size $D \times D$, in total $3 \times D \times D$.

Attention weight is given by $\text{Softmax}[K^TQ]$, where $K^T$ is $N \times D$, so we get $N \times N$. **Note that these are NOT learnable**, they are simply the result of taking the softmax of the matrix multiplication between K and Q.

The last sentence is a bit vague, but it sounds like the author wants us to compare the weights of a fully connected network of one layer? If this is the case then the total number of weights including the biases is given by $(DN)^2 + DN = D^2N^2 + DN$, since each neuron in the fully connected layer must be connected to the entire dimension of the input. And this grows $\mathcal{O}(N^2)$ with the sequence length N. Compare this to the self-attention that requires in total $3D^2 + 3D$ weights, which is clearly independent from the sequence length as far as learnable weights go (attention matrix is $N \times N$, so grows $\mathcal{O}(N^2)$ with seq length N, but doesn't contain any learnable weights), so it only scales quadratically $\mathcal{O}(D^2)$ w.r.t D and not N.


## 12.2
```
Why might we want to ensure that the input to the self-attention mechanism is the same size as the output? 
```

I think the simple answer is to make the attention weights meaningful to the value matrix, since the attention matrix is derived from Q and K that in turn creates relative similarities between the inputs.


## 12.3
```
Show that the self-attention mechanism (equation 12.8) is equivariant to a permutation XP of the data X, where P is a permutation matrix. In other words, show that: 

                        Sa[XP] = Sa[X]P
```

## 12.4
```
Consider the softmax operation: yi = softmaxi[z] = exp[zi] �5 j=1 exp[zj] , (12.19) in the case where there are five inputs with values: z1 = −3, z2 = 1, z3 = 100, z4 = 5, z5 = −1. Compute the 25 derivatives, ∂yi/∂zj for all i, j ∈ {1, 2, 3, 4, 5}. What do you conclude? 
```

# Bishop exercises chapter 12 - NLP
## 12.1
