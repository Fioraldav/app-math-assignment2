import numpy as np


def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)

    # Step 1: Calculate the differences
    differences = pred - tar

    # Step 2: Square the differences
    squared_differences = differences ** 2

    # Step 3: Calculate the mean of the squared differences
    mean_squared_differences = np.mean(squared_differences)

    # Step 4: Take the square root of the mean
    rmse = np.sqrt(mean_squared_differences)

    return rmse