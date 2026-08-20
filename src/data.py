"""Token-array plumbing for the training loop.

§5.5 requires the corpus to be memory-mapped and never read into RAM. That is
straightforward for one array, but Task 1 encodes the training split as two
files (the corpus ships as `TinyStoriesV2-GPT4-train.txt` and
`-train-part2.txt`), and the training loop wants one flat sequence of tokens to
sample windows from.

`ConcatTokens` is that one sequence: a read-only view over several memmaps that
indexes as though they were concatenated. np.concatenate would materialise
1.1GB of uint16 in RAM and defeat the point.
"""
import numpy as np


class ConcatTokens:
    """Several token arrays addressed as one, without copying them.

    Supports what the batching and evaluation paths actually use: `len()`,
    integer indexing, and slicing - including a slice that straddles the
    boundary between two parts, which is the only interesting case.

    params:
        parts: token arrays (memmaps are the point, but any sequence works)
    """

    def __init__(self, parts):
        self.parts = [p for p in parts if len(p) > 0]
        if not self.parts:
            raise ValueError("ConcatTokens needs at least one non-empty part")
        self.lengths = [len(p) for p in self.parts]
        # Cumulative start offset of each part, plus the total at the end.
        self.offsets = np.cumsum([0] + self.lengths)
        self.dtype = self.parts[0].dtype

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError("ConcatTokens supports contiguous slices only")
            return self._slice(start, stop)

        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        part = int(np.searchsorted(self.offsets, key, side="right") - 1)
        return self.parts[part][key - self.offsets[part]]

    def _slice(self, start, stop):
        """The tokens in [start, stop), copied out of however many parts it spans."""
        if stop <= start:
            return np.empty(0, dtype=self.dtype)

        first = int(np.searchsorted(self.offsets, start, side="right") - 1)
        last = int(np.searchsorted(self.offsets, stop - 1, side="right") - 1)
        if first == last:
            # The common case by far: windows are short and parts are huge, so
            # a window almost never straddles a boundary.
            offset = self.offsets[first]
            return np.asarray(self.parts[first][start - offset:stop - offset])

        chunks = []
        for part in range(first, last + 1):
            lo = max(start, self.offsets[part]) - self.offsets[part]
            hi = min(stop, self.offsets[part + 1]) - self.offsets[part]
            chunks.append(np.asarray(self.parts[part][lo:hi]))
        return np.concatenate(chunks)

    def __array__(self, dtype=None, copy=None):
        """Materialise everything. Only for small arrays - this is what §5.5
        says not to do to the training corpus, and it is here so an accidental
        np.asarray() on a huge ConcatTokens is at least explicit in a traceback.
        """
        out = np.concatenate([np.asarray(p) for p in self.parts])
        return out.astype(dtype) if dtype is not None else out

    def __repr__(self):
        return (f"ConcatTokens({len(self.parts)} parts, {len(self):,} tokens, "
                f"{self.dtype})")


def load_token_parts(paths, mmap_mode="r"):
    """Memory-map each path and present them as one array (§5.5).

    A single path still comes back as a plain memmap rather than a wrapper,
    so the common case carries no indirection.
    """
    arrays = [np.load(p, mmap_mode=mmap_mode) for p in paths]
    return arrays[0] if len(arrays) == 1 else ConcatTokens(arrays)
