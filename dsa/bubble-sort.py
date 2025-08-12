from typing import List

def bubble_sort(numbers: List[int]) -> List[int]:
    for i in range(len(numbers) - 1):
        for j in range(len(numbers) - 1 - i):
            if numbers[j] > numbers[j+1]:
                numbers[j+1], numbers[j] = numbers[j], numbers[j+1]
    return numbers
        
if __name__ == '__main__':
    import random
    nums = [random.randint(0, 1000) for i in range(10)]
    print(bubble_sort(nums))
