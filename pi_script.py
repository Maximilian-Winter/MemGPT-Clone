
# pi_script.py
import math

def leibniz_pi(iterations):
    pi, sign = 0.0, 1
    index = 0
    for _ in range(iterations):
        pi += sign / (2 * (2 * index + 1))
        index += 1
        sign *= -1

    return pi * 4

if __name__ == "__main__":
    iterations = int(input("Enter the number of iterations: "))
    result = leibniz_pi(iterations)
    print(f"Calculated PI using {iterations} iterations: {result}")