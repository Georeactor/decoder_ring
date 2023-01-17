Interesting params in https://github.com/mapmeld/transformers/blob/main/src/transformers/generation_utils.py

is_constraint_gen_mode

Not yet supported:

- https://github.com/XiangLi1999/ContrastiveDecoding
- https://arxiv.org/abs/2104.05336

Partially supported:

- time control https://arxiv.org/abs/2203.11370 == https://github.com/rosewang2008/language_modeling_via_stochastic_processes

Issues:

- https://github.com/martiansideofthemoon/rankgen

There is a rankgen pip package, but it has its own encoder/decoder and requires from a set of t5 models.

Supported but should be linked:

https://huggingface.co/blog/introducing-csearch contrastive search


use in code: https://twitter.com/moyix/status/1606103262138011649
