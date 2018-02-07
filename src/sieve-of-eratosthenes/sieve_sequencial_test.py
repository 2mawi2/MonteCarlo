import unittest
import src.sieve_sequencial as sieve


class TestSieve(unittest.TestCase):
    def test_sieve_of_eratosthenes(self):
        self.assertEqual([2, 3, 5, 7], sieve.sieve_of_eratosthenes(0, 10))

    def test_works_with_range_not_starting_with_zero(self):
        self.assertEqual([11, 13, 17, 19], sieve.sieve_of_eratosthenes(10, 20))

    def test_sieve_of_eratosthenes_python(self):
        self.assertEqual([2, 3, 5, 7], sieve.sieve_of_eratosthenes_python(10))

    def test_works_with_range_not_starting_with_zero_python(self):
        self.assertEqual([2, 3, 5, 7, 11, 13, 17, 19], sieve.sieve_of_eratosthenes_python(20))
