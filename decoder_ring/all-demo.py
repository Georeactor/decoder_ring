import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import pytorch_lightning as pl  # for TimeControl

from decoders import (
    BasicWriter,
    BeamSearch,
    ContrastiveSearch,
    GreedyDecoder,
    RandomSampling,
    TimeControl,
    TypicalDecoder,
)
from timecontrol.decode import GPT2TimeLMHeadModel
from timecontrol.encode import BrownianBridgeSystem

## Writer test
basic = BasicWriter("gpt2", RandomSampling)
writer_output = basic.write_text(
    prompt="Hello, my name is",
    max_length=20,
)
print(writer_output)

## HuggingFace tokenized string
# model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
content = tokenizer.encode("Hello, my name is", return_tensors="pt")

## Decoder tests
# decoder1 = GreedyDecoder(model)
# greedy_output = decoder1.generate_text(
#     prompt=content,
#     max_length=20,
# )
# txtop = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
# print(txtop)

# without random_seed, or set_random_seed, this will call logging.warn
# decoder2 = RandomSampling(model, random_seed=603)
# sampling_output = decoder2.generate_text(
#     prompt=content,
#     max_length=20,
# )
# txtop2 = tokenizer.decode(sampling_output[0], skip_special_tokens=True)
# print(txtop2)
#
# decoder3 = TypicalDecoder(model, random_seed=603, typical_p=0.4)
# typical_output = decoder3.generate_text(
#     prompt=content,
#     max_length=20,
# )
# txtop3 = tokenizer.decode(typical_output[0], skip_special_tokens=True)
# print(txtop3)
#
# decoder4 = ContrastiveSearch(model, random_seed=603, penalty_alpha=0.4, top_k=4)
# contrastive_output = decoder4.generate_text(
#     prompt=content,
#     max_length=20,
# )
# txtop4 = tokenizer.decode(contrastive_output[0], skip_special_tokens=True)
# print(txtop4)
#
# decoder5 = BeamSearch(model, early_stopping=True, num_beams=3)
# beam_output = decoder5.generate_text(
#     prompt=content,
#     max_length=20,
# )
# txtop5 = tokenizer.decode(beam_output[0], skip_special_tokens=True)
# print(txtop5)

tc_model = GPT2TimeLMHeadModel.from_pretrained("gpt2")
trainer = pl.Trainer(
    # gpus=1,
    max_epochs=1,
    min_epochs=1,
)
trainer.fit(
    BrownianBridgeSystem(
        # params via https://github.com/rosewang2008/language_modeling_via_stochastic_processes/blob/main/language_modeling_via_stochastic_processes/config/encoder/brownian.yaml
        {
            "data_params": {
                # "k": 5,
                "name": "recipe",
                "path": "~/Downloads/recipe/",
            },
            "loss_params": {
                "name": "simclr",
            },
            "model_params": {
                "eps": 1e-6,
                "hidden_size": 128,
                "latent_dim": 32,
                "name": "gpt2",
            },
            "optim_params": {
                "batch_size": 32,
                "learning_rate": 0.0001,
                "momentum": 0.9,
            },
        }
    )
)
trainer.save("./tcencoder")

decoder6 = TimeControl(tc_model, encoder=trainer.model, random_seed=22)
tc_output = decoder6.generate_text(
    prompt=content,
    max_length=20,
)
txtop6 = tokenizer.decode(tc_output[0], skip_special_tokens=True)
print(txtop6)
