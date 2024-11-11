import torch
import torch.nn.functional as F

def max_pool_grad_grad(input, grad_output, kernel_size, stride, padding, ceil_mode=False):

    input.requires_grad_(True)
    output, indices = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True, ceil_mode=ceil_mode)

    grad_output.requires_grad_(True)
    grad_input = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=grad_output, create_graph=True)[0]

    gradgrad1 = torch.zeros_like(input)
    gradgrad2 = torch.zeros_like(grad_output)

    return grad_input, gradgrad1, gradgrad2

# Example usage
input = torch.randn(1, 1, 4, 4, requires_grad=True)
grad_output = torch.randn(1, 1, 2, 2, requires_grad=True)

kernel_size = 2
stride = 2
padding = 0

grad_input, gradgrad1, gradgrad2 = max_pool_grad_grad(input, grad_output, kernel_size, stride, padding)
