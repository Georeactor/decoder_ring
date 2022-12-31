# Decoder_Ring

`pip install decoder-ring`

`from decoder_ring import ContrastiveSearch`

## Concept

The fluency and usefulness of text generation models depends on the decoder used to select tokens from probabilities and build the text output.

Two examples: greedy decoding always selects the most probable token; random sampling considers all possible tokens with their given probability.

The goal of decoder_ring is a common API with type hints, helpful error messages and logs, parameter restrictions, encouragement of random seeds, etc. to make text decoding clear and reproducible. In the future this should support many more decoder types.

## Documentation

I would like to expand on the documentation in all of the decoder options, links to relevant papers etc., to make this library and the overall decoder concept accessible to new users.

### Supported methods

- ContrastiveSearch (params: random_seed, penalty_alpha, top_k)
- GreedyDecoder
- RandomSampling (params: random_seed)
- TypicalDecoder (params: random_seed, typical_p)

### Writer Examples (text input and output)

```python
from decoder import BasicWriter, RandomSampling

basic = BasicWriter('gpt2', RandomSampling)
writer_output = basic.write_text(
    prompt="Hello, my name is", max_length=20, early_stopping=True
)
```

### Decoder Examples (with customization)

Start with a HuggingFace Transformers / PyTorch model and tokenized text:

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
content = tokenizer.encode("Hello, my name is", return_tensors="pt")
```

Example with Transformers' default greedy decoder:

```python
decoder1 = GreedyDecoder(model)
greedy_output = decoder1.generate_text(
    prompt=content, max_length=20, early_stopping=True
)
tokenizer.decode(greedy_output[0], skip_special_tokens=True)
```

Example with typical decoding, which will require a `random_seed` before generating text, and a `typical_p` between 0 and 1:

```python
decoder3 = TypicalDecoder(model, random_seed=603, typical_p=0.4)
typical_output = decoder3.generate_text(
    prompt=content, max_length=20, early_stopping=True
)

# new random seed
decoder3.set_random_seed(101)
typical_output_2 = decoder3.generate_text(
    prompt=content, max_length=20, early_stopping=True
)
```

## License

Apache license for compatibility with the Transformers library