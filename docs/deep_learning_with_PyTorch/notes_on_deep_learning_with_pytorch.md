# Notes On Deep Learning With PyTorch

[TOC]

## Introducing deep learning and the PyTorch library

### What is PyTorch

Python library that enables to build deep learnig projects. The main difference is that allows deep learning models to be expressed in a pythonic way. It is based on the core data structure of ```Tensors``` wich are similar to NumPy arrays. 

### What is this book ?

Is a starting point for software engineers, data scientists, and motivated students who are fluent in Python and want to become comfortable using PyTorch to build deep learning project.

### Why PyTorch?

	1. Simplicity: Using the library generally feels familiar to developers who have used Python previously.
	2. Similarity to NumPy: PyTorch feels like NumPy, but with GPU acceleration and auto- matic computation of gradients.
	3. Expressivity: Allows the developer to implement complicated models without undue complexity being imposed by the library.
	4. Can be used in production: Has a compelling story for the transition from research and development to production.

#### The deep learning revolution

With Machine learning, a data scientist is busy defining engineering features and feeding them to a learning algorithm. The results of the task will be as good as the features he engineers. On the right side of the figure, with deep learning, the raw data is fed to an algorithm that extracts hierarchical features automatically, based on optimizing the performance of the algorithm on the task. The results will be as good as the practitioner’s ability to drive the algorithm toward its goal.

![Comparison between Machine Learning and Deep Learning](./images/comparison_ml_vs_dl.png)

#### Immediate versus deferred execution

Immediate execution is useful because if problems arise in executing the expression, the Python interpreter, debugger, and similar tools have direct access to the Python objects involved.

```python
a = 30
b = 40
c = (a**2 + b**2) ** 0.5
```

Deferred execution means that most exceptions are be raised when the function is called, not when it’s defined.

```python
a = 30
b = 40

def pythagorean_expression(c1, c2):
	return (a**2 + b**2) ** 0.5

c = pythagorean_expression(a, b)
```

Things get tricky when specialized classes that have heavy operator overloading are used:

```python
a = InputParameterPlaceholder()
b = nputParameterPlaceholder()

def pythagorean_expression(c1, c2):
	return (a**2 + b**2) ** 0.5

c = pythagorean_expression(a, b)
```

Often in libraries that use this form of function definition, ompile the expression into a static computation graph (a graph of basic operations) that has some advantage over pure Python, such as improving performance. 

The fact that the computation graph is built in one place and used in another makes debugging more difficult. static graphs usually don’t mix well with standard Python flow control: they’re de-facto domain-specific languages implemented on top of a host language.

PyTorch that use immediate execution differ from deferred-execution frameworks, even though the underlying math is the same for both types.

The fundamental building block of a neural network is a neuron. Neurons are strung together in large numbers to form the network. The typical ecuation of a neuron is ```output = Tanh(wx + b)```

0. x is the input to a single neuron

1. w and b are the parameters or weights of the neuron and can be changed as needed.

2. To update the weights we assign error to each of the weights via backpropagation and then tweak the weights accordingly. 

3. Backpropagation requires computing the gradient of the output with respect to the weights.

4. We use automatic differentiation to compute the gradient automatically, saving us the trouble of writing the calculations by hand.

![Static graph for a simple computation corresponding to a single neuron](./images/static_graph.png)

The neuron gets compiled into a symbolic graph in which each node rep- resents individual operations (second row), using placeholders for inputs and out- puts. Then the graph is evaluated numerically (third row) when concrete numbers are plugged into the placeholders.

The gradient of the output with respect to the weights is constructed sym- bolically by automatic differentiation, which traverses the graph backward and multi- plies the gradients at individual nodes (fourth row). The corresponding mathematical expression is shown in the fifth row.

One of the major competing deep learning frameworks is **TensorFlow**, which has a graph mode that uses a similar kind of deferred execution. PyTorch sports a define-by-run dynamic graph engine in which the computation graph is built node by node as the code is eagerly evaluated.

![Dynamic graph for a simple computation corresponding to a single neuron](./images/static_graph.png)

The computation is broken into individual expressions, which are greed- ily evaluated as they’re encountered. The program has no advance notion of the inter- connection between computations. The bottom half of the figure shows the behind-the- scenes construction of a dynamic computation graph for the same expression. The expression is still broken into individual operations, but here those operations are eagerly evaluated, and the graph is built incrementally. Automatic differentiation is achieved by traversing the resulting graph backward, similar to static computation graphs.

Note that this does not mean dynamic graph libraries are inherently more capable than static graph libraries, just that it’s often easier to accomplish looping or condi- tional behavior with dynamic graphs. Dynamic graphs can change during successive forward passes. Different nodes can be invoked according to conditions on the outputs of the preceding nodes, for example, without a need for such conditions to be represented in the graph itself—a dis- tinct advantage over static graph approaches.

#### The deep learning landscape

0. **Theano** (University of Montreal) : 

	* One of the first deep learning frameworks, has ceased active development.

1. **TensorFlow** (Google):

	* Consumed Keras, promoting it to a first-class API
	* Provided an immediate execution eager mode (a new define-by-run API, increasing the library’s flexibility)

2. **PyTorch** (Facebook):

	* Consumed Caffe2 for its backend
	* Replaced most of the low-level code reused from the Lua-based Torch project.
	* Added a delayed execution graph mode runtime called TorchScript

**Summary**

TensorFlow has a robust pipeline to production, an extensive industrywide community, and massive mindshare. 

PyTorch has made huge inroads with the research and teaching community, thanks to its ease of use, and has picked up momentum as researchers and graduates train students and move to industry.

### PyTorch has the batteries included

PyTorch is a library that provides multidimensional arrays, called tensors in PyTorch parlance, and an extensive library of operations on them is provided by the torch module. Both tensors and related operations can run on the CPU or GPU.

![Basic high-level structure of a PyTorch project, with data loading, training, and deployment to production](./images/PyTorch_project_structure.png)

What do you need: A source of training data, an optimizer to adapt the model to the training data, and a way to get the model and data to the hardware that will be performing the calculations needed for training the model.

Utilities for data loading and handling can be found in ```torch.util.data```. The two main classes you’ll work with are ```Dataset```, which acts as the bridge between your custom data,  and a standardized PyTorch ```Tensor```. 

PyTorch also provides a deferred execution model named ```TorchScript```. Serialize a set of instruc- tions that can be invoked independently from Python. this execution mode gives PyTorch the opportunity to Just in Time (JIT) transform sequences of known operations into more efficient fused operations.
