import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimentions is a tuple of integers.
    Hint: use torch.ones and multiply by val, or use torch.zeros and add val.
    e.g. if dimensions = (2, 3), and val = 3, then the returned tensor should be of shape (2, 3)
    specifically, it should be:
    tensor([[3., 3., 3.], [3., 3., 3.]])
    """
    res = torch.ones(dimensions) * val
    return res


def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    res = A * B  # TODO: implement this function
    return res


def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i})
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.

    """
    res = torch.matmul(X, W.T)  # TODO: implement this function
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. ( sum {x_i * w_i}) and add the bias.
    Note that the dimensions of X and W should be compatible for multiplication.
    e.g: if X is a tensor of shape (1,3) then W could be a tensor of shape (N,3) i.e: (1,3) or (2,3) etc. but in order for
         matmul to work, we need to multiply by W.T (W transpose) so that the `inner` dimensions are the same.
    Hint: use torch.matmul to calculate the product.
          This allows us to use a batch of inputs, and not just a single input.
          Also, it allows us to use the same function for a single neuron or multiple neurons.
       """
    res = torch.matmul(X, W.T) + b  # TODO: implement this function
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    Hint: use PyTorch `heaviside` function.
    """
    res = torch.heaviside(sum_total, torch.tensor([0.0]))  # TODO: implement this function
    return res

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    Hint: use the functions you implemented above.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    output = calculate_activation(sum_total)  # TODO: implement this function
    return res


# Example usage of each function
if __name__ == "__main__":
    # create_tensor_of_val example
    print(create_tensor_of_val((2, 3), 3))
    # Output: tensor([[3., 3., 3.], [3., 3., 3.]])

    # calculate_elementwise_product example
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    print(calculate_elementwise_product(A, B))
    # Output: tensor([[ 5, 12], [21, 32]])

    # calculate_matrix_product example
    X = torch.tensor([[1, 2, 3]])
    W = torch.tensor([[4, 5, 6], [7, 8, 9]])
    print(calculate_matrix_product(X, W))
    # Output: tensor([[32, 50]])

    # calculate_matrix_prod_with_bias example
    b = torch.tensor([1, 2])
    print(calculate_matrix_prod_with_bias(X, W, b))
    # Output: tensor([[33, 52]])

    # calculate_activation example
    sum_total = torch.tensor([-1, 0, 1, 2])
    print(calculate_activation(sum_total))
    # Output: tensor([0., 0., 1., 1.])

    # calculate_output example
    print(calculate_output(X, W, b))
    # Output: tensor([[1., 1.]])