from typing import List

def sort(numbers: List[int]) -> List[int]:
    for i in range(len(numbers) - 1):
        j = i
        while j >= 0:
            if numbers[j] > numbers[j+1]:
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
            j -= 1
    return numbers
        
if __name__ == '__main__':
    import random
    nums = [random.randint(0, 1000) for i in range(10)]
    print(sort(nums))
