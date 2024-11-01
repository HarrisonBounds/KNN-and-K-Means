import numpy as np


def euclidean(a, b):
    """Returns the Euclidean distance between vectors a and b"""
    dist = np.sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)
    return dist

# returns Cosine Similarity between vectors a and b


def cosim(a, b) -> float:
    # Change to vectors
    a = np.array(a)
    b = np.array(b)

    numerator = np.dot(a, b)
    denominator = np.sqrt(np.sum(a**2)) / np.sqrt(np.sum(b**2))

    if numerator == 0 or denominator == 0:
        return 0
    # Generalize to higher dimensions
    return numerator / denominator


def in_same_dimension(a: np.ndarray, b: np.ndarray) -> bool:
    """ Determines if the two given vectors are in the 
        same dimension or not

    Args:
        a (np.ndarray): The first vector of any dimension
        b (np.ndarray): The second vector of any dimension

    Returns:
        bool: Whether the 2 vectors are the same dimension or not
    """
    return np.shape(a) == np.shape(b)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """ Returns the Hamming distance between vectors a and b

    Args:
        a (np.ndarray): A vector of any dimension
        b (np.ndarray): A vector of any dimension

    Returns:
        int: The Hamming distance between the two vectors

    Raises:
        ValueError: If the given vectors are different dimensions
    """
    # Ensure that the two vectors occupy the same dimension
    if not in_same_dimension(a, b):
        print(
            f"Given vectors have different shapes: " +
            f"{np.shape(a)} != {np.shape(b)}"
        )
        raise ValueError("Hamming requires 2 identically-shaped vectors")
    # Create a vector of boolean values where True indicates a difference
    comparison_vector = np.array([ai != bi for ai, bi in zip(a, b)])
    return comparison_vector.sum()
