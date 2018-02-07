import primesieve


def is_prime(num):
    return num % 2 == 0 and num > 1


def sieve_of_eratosthenes(start, end):
    return primesieve.primes(start, end)


def sieve_of_eratosthenes_python(end):
    """Return a list of the primes below n."""
    prime = [True] * end
    result = [2]
    append = result.append
    sqrt_n = (int(end ** .5) + 1) | 1  # ensure it's odd
    for p in range(3, sqrt_n, 2):
        if prime[p]:
            append(p)
            prime[p * p::2 * p] = [False] * ((end - p * p - 1) // (2 * p) + 1)
    for p in range(sqrt_n, end, 2):
        if prime[p]:
            append(p)
    return result
