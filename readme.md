# why?

1. I want to better learn Pytorch and how it works so what better way than to just re-implement some of its core features.

2. I write mainly in Go and haven't come across a lot of ML support in Go

3. I'd rather have a Go ML service instead of spinning up additional infrastructure to just support a python ML service in my Go projects

4. Go's static typing, native concurrency (avoid GIL problem in python), efficient memory management, single binary deployment and more make it a better interface compared to python IMO

Knowing the Pytorch is mainly implemented in c++ and c under the covers, I'm not expecting any performance gains by porting it to Go. But still, will be interesting to see how it compares.

# roadmap

- ~~Tensor Operations: addition, subtraction, multiplication, division,~~ and more advanced operations like dot products, transposition, and reshaping.

- Automatic Differentiation: basic version of autograd (forward & back propogation)

- Neural Network Layers: some sort of basic neural network layers (fully connected (dense) layers, convolutional layers, and recurrent layers)

- ~~Activation Functions: ReLU, Sigmoid,Tanh and softmax~~

- ~~Loss Functions: mean squared error (MSE),and binary cross-entropy, categorical cross entropy loss~~

- Optimizers: Stochastic Gradient Descent (SGD), Adam, and RMSprop

- ~~Data Handling: loading, save~~

- Model Definition and Training Loop: A framework to define models, compile them with loss functions and optimizers, and a training loop to fit models to data.

- GPU Support ???????

- Serialization: saving and loading models and their weights. Probably just model state and weights and not the entire model. [Reference](https://github.com/pytorch/pytorch/blob/761d6799beb3afa03657a71776412a2171ee7533/docs/source/notes/serialization.rst)
