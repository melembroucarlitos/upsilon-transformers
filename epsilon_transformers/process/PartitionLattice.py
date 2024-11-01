from typing import List


def generate_set_partitions(n: int) -> List[List[int]]:
    """
    Generate all possible partitions of a set of size n using an iterative approach.
    Returns a list of partitions, where each partition is a list of subsets.
    Each subset is represented as a list of integers from 0 to n-1.
    
    Example for n=3:
    [
        [[0,1,2]],           # One subset containing all elements
        [[0,1],[2]],         # Two subsets
        [[0,2],[1]],         # Two subsets
        [[1,2],[0]],         # Two subsets
        [[0],[1],[2]]        # Three singleton subsets
    ]
    """
    if n == 0:
        return []
    
    # Start with the partition of the set {0}
    partitions = [[[0]]]
    
    # For each new element k, extend all existing partitions
    for k in range(1, n):
        new_partitions = []
        
        for partition in partitions:
            # Add k to each existing subset
            for i in range(len(partition)):
                new_partition = [list(subset) for subset in partition]
                new_partition[i].append(k)
                new_partitions.append(new_partition)
            
            # Create new singleton subset {k}
            new_partition = [list(subset) for subset in partition]
            new_partition.append([k])
            new_partitions.append(new_partition)
        
        partitions = new_partitions
    
    return partitions