from typing import List

def insertion_sort(numbers: List[int]) -> List[int]:
    for i in range(len(numbers) - 1):
        j = i
        while j >= 0:
            if numbers[j] > numbers[j+1]:
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
            j -= 1
    return numbers

def sort(numbers: List[int]) -> List[int]:
    max_num = max(numbers)
    len_num = len(numbers)
    size = max_num // len_num
    buckets = [[] for _ in range(size)]
    for num in numbers:
        i = num // size
        if i >= size:
            buckets[size-1].append(num)
        buckets[i].append(num)
    
    for i in range(size):
        buckets[i] = insertion_sort(buckets[i])
    
    result = []
    for i in range(size):
        result += buckets[i]
    return result
    
        
if __name__ == '__main__':
    import random
    nums = [random.randint(0, 1000) for i in range(10)]
    print(sort(nums))
