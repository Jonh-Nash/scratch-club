from typing import List

def sort(numbers: List[int]) -> List[int]:
    for i in range(len(numbers)):
        min_num_index = i
        for j in range(i, len(numbers)):
            if numbers[min_num_index] > numbers[j]:
                min_num_index = j
        numbers[i], numbers[min_num_index] = numbers[min_num_index], numbers[i]
        
    return numbers
        
if __name__ == '__main__':
    import random
    nums = [random.randint(0, 1000) for i in range(10)]
    print(sort(nums))
