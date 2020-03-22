from degmo import make_dataset, add_dataset

from .bair_push import load_bair_push, load_bair_push_seq

add_dataset('bair_push', load_bair_push)
add_dataset('bair_push_seq', load_bair_push_seq)