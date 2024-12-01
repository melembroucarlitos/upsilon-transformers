from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, cast
from types import FrameType
import inspect
from regex import B
from jaxtyping import Float
from transformer_lens import HookedTransformer

from epsilon_transformers.process.Process import Process

# TODO: Write test for PROCESS_REGISTRY
# TODO: Think if you really need PROCESS_REGSITRY (if only getting called during dataloader creation, it may be better to have the dataloader take in a process)
# TODO: Add test to make sure that all members of this module are a member of Process
# TODO: Find paper where mess3 process is introduced
# TODO: Think through whether self.name is necessary (review it's usage in derive_mixed_state_presentation)
# TODO: Move _create_hmm into the init function prior to super()__init__()


class ZeroOneR(Process):
    def __init__(self, prob_of_zero_from_r_state: float = 0.5):
        self.name = "z1r"
        self.p = prob_of_zero_from_r_state
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        state_names = {"0": 0, "1": 1, "R": 2}
        T[0, state_names["0"], state_names["1"]] = 1.0
        T[1, state_names["1"], state_names["R"]] = 1.0
        T[0, state_names["R"], state_names["0"]] = self.p
        T[1, state_names["R"], state_names["0"]] = 1 - self.p

        return T, state_names


class RRXOR(Process):
    def __init__(self, pR1=0.5, pR2=0.5):
        self.name = "rrxor"
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 5, 5))
        state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
        T[0, state_names["S"], state_names["0"]] = self.pR1
        T[1, state_names["S"], state_names["1"]] = 1 - self.pR1
        T[0, state_names["0"], state_names["F"]] = self.pR2
        T[1, state_names["0"], state_names["T"]] = 1 - self.pR2
        T[0, state_names["1"], state_names["T"]] = self.pR2
        T[1, state_names["1"], state_names["F"]] = 1 - self.pR2
        T[1, state_names["T"], state_names["S"]] = 1.0
        T[0, state_names["F"], state_names["S"]] = 1.0

        return T, state_names


class Mess3(Process):
    def __init__(self, x=0.15, a=0.6):
        self.name = "mess3"
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names

class Even(Process):
    def __init__(self):
        self.name = "Even"
        super().__init__()

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5   # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5   # From state 0, emit 1, go to state 1
        T[1,1,0] = 1.0   # From state 1, emit 1, go to state 0
        return T, state_names

class GoldenMean(Process):
    def __init__(self):
        self.name = "Golden"
        super().__init__()

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5  # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5  # From state 0, emit 1, go to state 1
        T[0,1,0] = 1.0  # From state 1, emit 0, go to state 0 
        return T, state_names
 
class TransitionMatrixProcess(Process):
    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix
        super().__init__()

    def _create_hmm(self):
        return self.transition_matrix, {
            i: i for i in range(self.transition_matrix.shape[0])
        }


PROCESS_REGISTRY: dict[str, type] = {
    key: value
    # cast because we know the current frame has the above classes
    for key, value in cast(FrameType, inspect.currentframe()).f_locals.items()
    if isinstance(value, type) and issubclass(value, Process) and key != "Process"
}

def _generate_all_possible_n_length_sequences(vocab_len: int, context_len: int) -> List[List[int]]:
    result = [[]]
    for _ in range(context_len):
        result = [s.append(c) for s in result for c in range(vocab_len)]
    return result

def _generate_all_sequences_up_to_length(max_len: int, vocab_len: int) -> List[List[int]]:
    """Generate all possible sequences of length 0 to max_len"""
    all_strings = [[]]
    for length in range(max_len):
        all_strings.extend([s + [c] for s in all_strings[-vocab_len**length:] for c in range(vocab_len)])
    return all_strings[1:] # Omit empty sequence

def _find_subsequence_idx(lst_of_sequences: List[List[int]], subsequence: List[int]) -> int:
    """Find the first full sequence that starts with the subsequence"""
    subseq_len = len(subsequence)
    for i, full_seq in enumerate(lst_of_sequences):
        if full_seq[:subseq_len] == subsequence:
            return i
    raise ValueError(f"Subsequence {subsequence} not found at start of any sequence")

@dataclass
class Glut:
    vocab_len: int
    context_len: int
    table: Dict[str, Float[torch.Tensor, "vocab_len"]]

    def __init__(self, vocab_len: int, context_len: int, all_possible_n_length_sequences: List[List[int]], probs: Float[torch.Tensor, "num_n_length_strings ctx_len vocab_len"]) -> Dict[str, Float[torch.Tensor, "vocab_len"]]:
        assert len(all_possible_n_length_sequences) == probs.shape[0], f"Number of sequences ({len(all_possible_n_length_sequences)}) must match first dimension of probs ({probs.shape[0]})"
        assert vocab_len < 10, "current implementation only deals w/ single digit state enumerations"

        self.vocab_len = vocab_len
        self.context_len = context_len

        all_sequences_up_to_ctx_len = _generate_all_sequences_up_to_length(max_len=context_len, vocab_len=vocab_len)
        glut_dict = dict()
        for seq in all_sequences_up_to_ctx_len:
            seq_idx = _find_subsequence_idx(lst_of_sequences=all_possible_n_length_sequences, subsequence=seq)
            glut_dict[''.join([str(x) for x in seq])] = probs[seq_idx, len(seq) - 1]
        self.table = glut_dict

def _create_hmm_glut(glut: Glut) -> Float[np.ndarray, "vocab_len num_snum_possible_strs num_possible_strs"]:
    states = list(glut.table.keys())
    num_states = len(states)

    hmm = np.zeros((glut.vocab_len, num_states, num_states))
    for i, from_state in enumerate(states):
        probs = glut.table[from_state]
        for j, to_state in enumerate(states):
            if to_state[:-1] == from_state:
                token = int(to_state[-1])
                hmm[token, i, j] = probs[token]
    return hmm

def _minify_hmm_glut(hmm: Float[np.ndarray, "vocab_len num_possible_strs num_possible_strs"]) -> Float[np.ndarray, "vocab_len num_states num_states"]:
    raise NotImplementedError

def transformer_to_hmm(model: HookedTransformer, batch_size: int, minify: bool=True) -> Process:
    all_possible_transformer_strings = _generate_all_possible_n_length_sequences(vocab_len=model.cfg.d_vocab, context_len=model.cfg.n_ctx)
    all_possible_transformer_strings_batched = [all_possible_transformer_strings[i:i + batch_size] for i in range(0, len(all_possible_transformer_strings), batch_size)]

    logits_batched: List[Float[torch.Tensor, "batch_size ctx_len d_vocab"]] = [model(torch.tensor(batch)) for batch in all_possible_transformer_strings_batched]
    logits_flattened = torch.cat(logits_batched, dim=0)
    probs = F.softmax(logits_flattened, dim=-1)

    glut = Glut(vocab_len=model.cfg.d_vocab, context_len=model.cfg.n_ctx, all_possible_n_length_strings=all_possible_transformer_strings, probs=probs)
    hmm = _create_hmm_glut(glut=glut)
    if minify:
        hmm = _minify_hmm_glut(hmm=hmm)
    return TransitionMatrixProcess(transition_matrix=hmm)

if __name__ == "__main__":
    print(_generate_all_sequences_up_to_length(max_len=5, vocab_len=4))