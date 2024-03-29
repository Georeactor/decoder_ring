from transformers import AutoModelForCausalLM, AutoTokenizer

from decoders import (
    BasicWriter,
    BeamSearch,
    ContrastiveSearch,
    GreedyDecoder,
    RandomSampling,
    TypicalDecoder,
)

## Writer test
basic = BasicWriter("gpt2", RandomSampling)
writer_output = basic.write_text(
    prompt="Hello, my name is",
    max_length=20,
)
print(writer_output)

## HuggingFace tokenized string
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
content = tokenizer.encode("Hello, my name is", return_tensors="pt")

## Decoder tests
decoder1 = GreedyDecoder(model)
greedy_output = decoder1.generate_text(
    prompt=content,
    max_length=20,
)
txtop = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
print(txtop)

# without random_seed, or set_random_seed, this will call logging.warn
decoder2 = RandomSampling(model, random_seed=603)
sampling_output = decoder2.generate_text(
    prompt=content,
    max_length=20,
)
txtop2 = tokenizer.decode(sampling_output[0], skip_special_tokens=True)
print(txtop2)

decoder3 = TypicalDecoder(model, random_seed=603, typical_p=0.4)
typical_output = decoder3.generate_text(
    prompt=content,
    max_length=20,
)
txtop3 = tokenizer.decode(typical_output[0], skip_special_tokens=True)
print(txtop3)

decoder4 = ContrastiveSearch(model, random_seed=603, penalty_alpha=0.4, top_k=4)
contrastive_output = decoder4.generate_text(
    prompt=content,
    max_length=20,
)
txtop4 = tokenizer.decode(contrastive_output[0], skip_special_tokens=True)
print(txtop4)

decoder5 = BeamSearch(model, early_stopping=True, num_beams=3)
beam_output = decoder5.generate_text(
    prompt=content,
    max_length=20,
)
txtop5 = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print(txtop5)
