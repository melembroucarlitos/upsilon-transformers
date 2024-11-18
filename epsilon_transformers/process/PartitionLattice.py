from typing import List
from jaxtyping import Float
import numpy as np

from epsilon_transformers.process.Process import Process
from epsilon_transformers.process.processes import TransitionMatrixProcess

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

def coarse_grain_process(hmm_tensor: Float[np.ndarray, "vocab_len num_states num_states"], state_groups) -> Process:
    # Verify input tensor is valid probability distribution
    assert np.allclose(np.sum(hmm_tensor, axis=(0,2)), 1), "Input tensor must sum to 1 over emissions and next states"
    assert np.all(hmm_tensor >= 0), "Input tensor must contain non-negative probabilities"
   
    emission_vocab, num_states, _ = hmm_tensor.shape
    num_new_states = len(state_groups)
   
    # Verify state groups are valid
    all_states = sorted([s for group in state_groups for s in group])
    assert len(all_states) == num_states, "State groups must contain each state exactly once"
    assert all_states == list(range(num_states)), "State groups must contain valid state indices"

    # Initialize new tensor
    new_tensor = np.zeros((emission_vocab, num_new_states, num_new_states))

    # For each emission
    for e in range(emission_vocab):
        # For each new state (source)
        for i, group_i in enumerate(state_groups):
            # Get total probability mass coming from states in group_i
            group_i_mass = np.sum([np.sum(hmm_tensor[e,old_i,:]) for old_i in group_i])
            
            if group_i_mass > 0:  # Avoid division by zero
                # For each new state (destination)
                for j, group_j in enumerate(state_groups):
                    # Sum probabilities from all states in group_i to all states in group_j
                    prob_sum = sum(hmm_tensor[e,old_i,old_j] 
                                    for old_i in group_i 
                                    for old_j in group_j)
                    # Normalize by total probability mass from group_i
                    new_tensor[e,i,j] = prob_sum / len(group_i)
    
    # Normalize to ensure it's still a valid probability distribution
    # Sum over emissions and next states should be 1 for each current state
    normalization = np.sum(new_tensor, axis=(0,2))
    for i in range(num_new_states):
        if normalization[i] > 0:  # Avoid division by zero
            new_tensor[:,i,:] /= normalization[i]

    return TransitionMatrixProcess(transition_matrix=new_tensor)

def generate_all_coarse_grained_processes(process: Process) -> List[Process]:
    partitions = generate_set_partitions(process.num_states)
    return [coarse_grain_process(process=process.transition_matrix, partition=partition) for partition in partitions]

if __name__ == "__main__":
    from epsilon_transformers.process.processes import ZeroOneR

    z1r = ZeroOneR()
    expected = np.array([[[.25, 0.5],
                          [0.0, 0.0]],
                         
                         [[.25, 0.0],
                          [1.0,0.0]]])
    actual = coarse_grain_process(hmm_tensor=z1r.transition_matrix, state_groups=[[0,2],[1]]).transition_matrix
    assert np.allclose(expected, actual)
    print(z1r.transition_matrix)