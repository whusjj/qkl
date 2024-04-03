import random

def generate_random_integers(low, high, m):
    return [random.randint(low, high) for _ in range(m)]