import torch as t
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
        # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with t.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(t.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(t.matmul(weight_mat, v), dim=0, eps=self.eps)
                sigma = t.dot(u, t.matmul(weight_mat, v))
                weight = weight / sigma
            return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, t.nn.Parameter(weight))
    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)
    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (t.nn.ConvTranspose1d,
                           t.nn.ConvTranspose2d,
                           t.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Standardize(nn.Module):
    '''Standardize the input to have mean 0 and std 1'''
    def __init__(self, epsilon=1e-7, dim=(2, 3), return_mean_and_std=False, **kwargs):
        super(Standardize, self).__init__()
        self.epsilon = epsilon
        self.dim = dim
        self.return_mean_and_std = return_mean_and_std


    def forward(self, x):

        # mean, variance = tf.nn.moments(x, axes=self.dim, keepdims=True)
        mean = t.mean(x, dim=self.dim, keepdim=True)
        var = t.var(x, dim=self.dim, keepdim=True)
        std = t.sqrt(var + self.epsilon)  # epsilon to avoid dividing by zero
        x_normed = (x - mean) / std
        if self.return_mean_and_std:
            return [x_normed, mean, std]
        return x_normed

    # def compute_output_shape(self, input_shape):
    #     if self.return_mean_and_std:
    #         return [input_shape, input_shape, input_shape]
    #     return input_shape

class Normalization(nn.Module):
    def __init__(self, ord='euclidean', eps=1e-7, **kwargs):
        # super().__init__(**kwargs)
        super(Normalization, self).__init__()
        self.ord = ord
        self.eps = eps


    def forward(self, x):
        shape = x.shape
        # expanded_size = [-1]
        # for i in range(1, b):
        #     expanded_size.append(1)
        norm = t.norm(x.flatten(1), p=2, keepdim=True).flatten(0)
        for i in range(len(shape) - 1):
            norm = norm.unsqueeze(-1)
        return x / norm#(K.reshape(tf.norm(K.batch_flatten(x), ord=self.ord, axis=1), expanded_size) + self.eps)
