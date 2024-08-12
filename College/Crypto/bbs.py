from random import randint
from math import gcd as bltin_gcd
import matplotlib.pyplot as plt
import numpy as np

# Utility Functions


def isPrime(number):
    if number == 1 or number == 2 or number == 3:
        return True
    if number == 4:
        return False

    index = 3
    while number > index:
        if number % index == 0:
            return False
        else:
            index += 1

    if index == number:
        return True


def isCongruentNumber(number):
    if (number - 3) % 4 == 0:
        return True
    else:
        return False


def coprime(a, b):
    return bltin_gcd(a, b) == 1


def findX(p, q):
    if isPrime(p) and isCongruentNumber(p) and isPrime(q) and isCongruentNumber(q):
        n = p * q
        x = 1
        while coprime(n, x):
            x = randint(0, n)
        return x


# Blum Blum Shub (BBS) Class


class BBS:
    p = 0
    q = 0
    n = 0
    seed = 0
    generatedValues = []

    def __init__(self, p, q):
        self.setP(p)
        self.setQ(q)
        if self.p > 0 and self.q > 0:
            self.__setN()
            self.__setSeed()

    def setP(self, p):
        if not self.__checkParams(p):
            self.p = p

    def setQ(self, q):
        if not self.__checkParams(q):
            self.q = q

    def __checkParams(self, number):
        isError = False
        if not isPrime(number):
            print(number, "is not prime")
            isError = True

        return isError

    def __setN(self):
        self.n = self.p * self.q

    def __setSeed(self):
        while not coprime(self.n, self.seed) and self.seed < 1:
            self.seed = randint(0, self.n - 1)

    def __generateValue(self):
        if self.p > 0 and self.q > 0:
            x = 0
            while not coprime(self.n, x):
                x = randint(0, self.n)
            return pow(x, 2) % self.n

    def generateBits(self, amount):
        if self.p == self.q:
            print("p should be different than q")
            return False

        if self.n == 0:
            print("N is equal to 0")
            return False

        else:
            bitsArray = []
            amount += 1

            for i in range(amount):
                generatedValue = self.__generateValue()
                self.generatedValues.append(generatedValue)

                if generatedValue % 2 == 0:
                    bitsArray.append(0)
                else:
                    bitsArray.append(1)

            return bitsArray


# Tests Class


class Tests:

    def singleBit(self, bits):
        minValue = 9725
        maxValue = 10275
        countBit1 = 0

        for bit in bits:
            if bit == 1:
                countBit1 += 1

        if countBit1 > minValue and countBit1 < maxValue:
            print("True. The sum of ones in a row is equal", countBit1)
            return True
        else:
            print("False. The sum of ones in a row is equal", countBit1)
            return False

    def series(self, bits):
        index = 0
        nextIndex = 1
        longestSeries = 0
        currentSeries = 1
        value = bits[0]
        breakSeries = False

        for i in bits:
            if nextIndex < len(bits):

                if value is bits[nextIndex]:
                    currentSeries += 1
                    if currentSeries > longestSeries:
                        longestSeries = currentSeries
                        value = i

                if value is not bits[nextIndex]:
                    currentSeries = 1
                    value = bits[nextIndex]

            index += 1
            nextIndex += 1

        return {"longestSeries": longestSeries, "value": value}

    def test_uniformity(self, samples: list[int]):
        """
        Test if the generated samples are uniformly distributed by plotting a histogram
        :param samples: the list of pseudo-random samples
        """
        plt.hist(samples, bins=20, density=True)
        plt.title("Uniformity Test - Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

    def test_scalability(self, samples: list[int]):
        """
        Test the scalability by plotting samples against their indices to check for patterns
        :param samples: the list of pseudo-random samples
        """
        plt.scatter(range(len(samples)), samples)
        plt.title("Scalability Test - Scatter Plot")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()

    def test_consistency(
        self, n_tests: int = 5, n_samples: int = 1000, p: int = 7, q: int = 31
    ):
        """
        Test the consistency by generating multiple sequences and comparing them
        :param n_tests: number of sequences to generate
        :param n_samples: number of samples in each sequence
        :param p: the prime number p for BBS
        :param q: the prime number q for BBS
        """
        sequences = [BBS(p, q).generateBits(n_samples) for _ in range(n_tests)]

        plt.figure(figsize=(12, 8))
        for i, seq in enumerate(sequences):
            plt.plot(seq, label=f"Sequence {i + 1}")

        plt.title("Consistency Test - Multiple Sequences")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Generate bits using BBS
    bbs = BBS(7, 31)
    bits = bbs.generateBits(20000)

    # Run Tests
    tests = Tests()
    print(tests.series(bits))
    print(tests.singleBit(bits))

    # Uniformity Test
    tests.test_uniformity(bits)

    # Scalability Test
    tests.test_scalability(bits)

    # Consistency Test
    tests.test_consistency()
