def derive(f, x, h=0.0001):
    # Calculate f(x + h) and f(x - h)
    f_x_plus_h = f(x + h)
    f_x_minus_h = f(x - h)

    # Apply the central difference formula
    derivative = (f_x_plus_h - f_x_minus_h) / (2 * h)

    return derivative