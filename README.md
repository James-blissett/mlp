# mlp
Building an multi-layer perceptron by hand to build my understanding of the fundamentals.

### What is a Multi-Layer Perceptron (MLP), and why the name?

The name comes from the 1950's when a guy called Frank Rosenblatt introduced the single-layer perceptron, then it was seriously picked up by Geffory Hinton in the 80's [1]. 

The fact that the name MLP was coined in the 50's is reason enough why it is called a perceptron, every second new invention back then was some sort of 'tron'.

## Definition of MLP
An MLP is a type of neural network. So, let's start by defining what a neural network is. A neural net is a collection of nodes with different 'strength' connections between eachother. The gains, or 'strength', of the connections between nodes are called weights, which are tuned during a training process to embed knowledge in the network. This knowledge is accessed when a signal, which would often be a numerical expression for some input sentence, is passed through the network of nodes.

An MLP is a neural net with a very particular structure. It has an input layer, an output layer, and some number 'n' of layers sandwiched between the input and output layers. The middle layers are called 'hidden' layers. 

**Why is it structured in a sandwich like this? Well:**
- the input layer is just the entry point for your data. It is shaped depending on what your input data looks like, it's 'shape'. 
- the output layer is determined by the shape of your desired output. If it's words then you'll need a layer with a node/neuron (they're used interchangably) for each element of your vocabulary (a vocab is often not expressed by words, but often by chunks of words + words, like 'ing')
- the middle layers
    - first, if you didn't have the middle, then you would be mapping the inputs directly to the outputs, which is a *linear* relationship!
    - we want to represent more complex things than just linear relationships, which is why we introduce more layers. Note that this can be achieved by one massive single layer representing all possibly necessary functions, or, as it is done in practice, many layers of neurons stacked in series. It turns out that if you stack the layers you need fewer neurons in total vs one enormous single layer. 

## Weights, Biases, and the difference between them
The network of nodes expresses knowledge and relationship by each connection between nodes having a weight, and each node itself having a bias. The weights we have already gone over, those are the strengths between different nodes. Thinking geometrically, they are the slope and tilt of the activation of the node. They are expressed below as $w_n$. The previous nodes in the layer of n nodes preceeding the node we're inspection are represented by $x_n$ in the below equation.

Biases are the node's baseline/threashold. If the nodes preceeding the node of focus are 0, then the neuron will still fire at the bias. Adding the bias allows you to shape the baseline behaviour of the network without any activations. The output takes the form: output = activation( w1*x1 + w2*x2 + ... + wn*xn + b ). Together, the weights shape the non-linear knowledge function we're trying to represent, and the biases move it 'up' and 'down', or whatever that measn in n dimensional space haha. 

The more proper algebraically defined version of the above equation is as follows, where $\sigma!$ is just a symbol used to represent the activation function. $\sigma!$ is the same as $f$ in $f(x)$, but is just *special* becasuse this is machine learning. 

$$y = \sigma!\left( \sum_{i=1}^{n} w_i x_i + b \right)$$

The pre-activation version is as follows. We'll come back to this later probably. 
$$z = \sum_{i=1}^{n} w_i x_i + b \qquad y = \sigma(z)$$

And it may also be useful to define the vector form of the equation too.
$$y = \sigma!\left( \mathbf{w}^\top \mathbf{x} + b \right)$$

## How information moves through the neural net
How does the network turn inputs into useful outputs?

### Step 1: Forward Pass
Firstly, the network needs to be initialised with a set of values to start refining from. These are chosen at random to ensure the network doesn't start off with any biases baked in, that could affect the training process. The starting value range is chosen between $(-1,1)$.

Then for each node, the pre-activation weighted sum is calculated. After that, the activation function is applied (the $\sigma!$). This activation function is often a Rectified Linear Unit (ReLU) function. What is a ReLU? It is a function that is positive linear when $x$ is positive, and is zero when $x$ is negative. Why apply this to the nodes? Well, at a high level an activation function is applied to allow positive values to pass unchanged and sets negative values to zero. 

<p align="center">
  <img src="image-3.png" alt="the shape of a ReLU and function" width="600">
</p>


But more specifically, the reason for applying the activation function is very cool. Suppose each layer just computes its pre-activation and passes it straight on:

$$z^{(1)} = W_1 x + b_1$$

$$z^{(2)} = W_2 z^{(1)} + b_2 = W_2(W_1 x + b_1) + b_2$$

Multiply that out:

$$z^{(2)} = (W_2 W_1)\, x + (W_2 b_1 + b_2)$$

Notice $W_2 W_1$ is just some matrix — call it $W'$ — and $W_2 b_1 + b_2$ is
just some vector $b'$. So the two-layer network is equivalent to:

$$z^{(2)} = W' x + b'$$

...a single linear layer. So, if you don't apply an activation function, your massive fancy *deep* neural network collapses into just 1 layer. The activation function is what enables a neural network to be *deep*. What you're doing, geometrically, is that by applying the non-linear activation function, you bend the space a little, meaning the stacked layers can't be represented as a linear combination and you can therefore represent useful non-linear features in your model. The way layers then connect together is shown in the below diagram.

<p align="center">
  <img src="image-2.png" alt="How Weights and Biases are used" width="500">
</p>

The ReLU function is used specifically (vs other functions such as a Sigmoid which is also often seen) because it has the following useful properties:

1) Sparse Representations: When z is negative, the ReLU is 0 - this is called a sparse representation because you will get some sparse field of values with lots of zeros in-between. On the other hand, a sigmoid will often produce some negative number when z is negative - this is call a dense representation because you will still get lots of non-zero values. It turns out sparse representations are better for learning, and I won't try answer why.
2) Reduces the vanishing gradient problem: for ReLU, as $|x|$ increases the gradient stays constant, whereas for a sigmoid as $|x|$ increases the gradient becomes increasingly smaller, causing issues for learning. We'll get into the specifics of gradients and all those other things later don't worry.

The activation function is then computed for each layer in sequence from layer 1 through to the second last layer. If you think that sounds slow then you're right, that's what diffusion models are for (though we won't get into them here).

Now for the final layer. Remember earlier that the final layer will take the shape of whatever you want to output. If the purpose of the MLP is a classification task, then the output of the MLP will be one value found using a sigmoid. This output of the sigmoid, for the classification task, will be some probability that the input belongs to a certain category, between 0 to 1. You can see this geometrically in the above diagram, where the sigmpod function has an asymptote at 0 and 1.

If this were an LLM, your final layer may have $50,000$ nodes, one for each vocab element. Your pre-activation score will be a vector $z = [z_1, z_2, \ldots, z_V] \quad (V = \text{vocabulary size})$ where each $z_n$ corresponds to one of the $50,000$ nodes. Each node pre-activation value $z_n$ is called a logit. Only the pre-activation scores for the final layer are called logits by the way. For an LLM, instead of a sigmoid you need something called a SoftMax function, which takes the massive pre-activation vector $z$ and turns it into a single probability distribution. We won't spend more time on it, other than saying it looks like this:

$$P(\text{token } i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

### Step 2: Loss Calculation
Finally on to step 2.

A loss function is a function that evaluates the difference between the model's output and the desired output; it is something we want to minimise. It is synonymous with the error function, or cost function which you may be familiar with if you have studied control theory.

<p align="center">
  <img src="image-1.png" alt="Loss Function Categorisation" width="600">
</p>

As seen in the above figure, there are two types of loss function: classification task loss functions, and regression task loss functions. Some examples of each type of loss include:

**Classification** (predicts a category):
- Spam detection — spam or not spam
- Image labelling — cat, dog, or bird
- Next-token prediction (LLMs) — which token comes next

**Regression** (predicts a continuous value):
- House price prediction — a dollar amount
- Temperature forecasting — tomorrow's temperature
- Robot joint control — target joint angle
 
We will restrict scope to classification tasks for now. The type of function used for classification tasks is called a cross-entropy loss function. We restrict scope further here to binary cross-entropy, for when you have to decide between a yes or no. The binary cross-entropy loss function takes the form:


For a single prediction:
 
$$L = -\big[\, y \log(p) + (1 - y)\log(1 - p) \,\big]$$
 
where $y$ is the true label (0 or 1) and $p$ is the predicted probability of
class 1 (the sigmoid output).
 
- If $y = 1$: reduces to $L = -\log(p)$
- If $y = 0$: reduces to $L = -\log(1 - p)$
Either way, it comes out to $-\log(\text{probability assigned to the correct class})$.
 
Averaged over $N$ examples, you get the cost (loss) function:
 
$$L = -\frac{1}{N}\sum_{n=1}^{N} \Big[\, y_n \log(p_n) + (1 - y_n)\log(1 - p_n) \,\Big]$$
 
Now we have a loss function, we need to figure out how to minimise it.

### Step 3: Backpropogation
Before we dive into backpropogation, let's first consider at a high level what we're doing here in order to minimise the loss function, and how we'll do it. The goal of training, as stated earlier, is to find the model weights that minimise the loss function. The process of minimising that loss function is called gradient descent, and backpropogation is used in gradient descent to find the gradient vector (which will be explained in a moment).

What is gradient descent? Imagine you are standing on the surface shown in the below figure. You want to 




<p align="center">
  <img src="image-5.png" alt="surface" width="500">
</p>