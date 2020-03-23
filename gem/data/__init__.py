from degmo import make_dataset, add_dataset

from .bair_push import load_bair_push, load_bair_push_seq
from .compressed import load_compressed_dataset

add_dataset('bair_push', load_bair_push)
add_dataset('bair_push_seq', load_bair_push_seq)
add_dataset('compressed', load_compressed_dataset)