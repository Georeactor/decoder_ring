## Decoder Magic

### Concept

The fluency and usefulness of text generation models depends on the decoder used to select tokens from probabilities and build the text output.

Greedy decoding always selects the most probable token, random sampling considers all possible tokens with their given probability.

We would like a common API with type hints, helpful error messages and logs, parameter and random seed restrictions, etc. to make the method of decoding clear and reproducible. In the future this should support many decoder types.

### Documentation

I would like to expand on the documentation in all of the decoder options to make this library and the overall decoder concept accessible to new users.

### Examples

Start with a HuggingFace Transformers / PyTorch model and tokenized text:

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
content = tokenizer.encode("Hello, my name is", return_tensors="pt")
```

Basic example with the default greedy decoder:

```
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

### License

Apache license for compatibility with the Transformers library