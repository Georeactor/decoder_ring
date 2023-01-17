import os

import torch
import transformers
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
from timecontrol.encode import BrownianBridgeSystem, GPT2OUEncoder

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

# based on https://github.com/rosewang2008/language_modeling_via_stochastic_processes/blob/main/language_modeling_via_stochastic_processes/scripts/train_encoder.py
tc_model = GPT2TimeLMHeadModel.from_pretrained("gpt2")
trainer = pl.Trainer(
    # gpus=1,
    max_epochs=1,
    min_epochs=1,
)
sys = BrownianBridgeSystem(
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
if os.path.isfile("./mlp.pt"):
    HIDDEN_DIM = 128

    def load_cl_model(filepath, latent_dim, base_model, use_section_ids, token_size):
        model = GPT2OUEncoder(
            model_name="gpt2",
            hidden_dim=HIDDEN_DIM,
            latent_dim=latent_dim,
            finetune_gpt2=False,
        )
        if use_section_ids:
            model.model.resize_token_embeddings(token_size)

        transformers.__spec__ = "gpt2"  # Avoid bug
        state_dict = torch.load(filepath)
        new_dict = {}
        for k, v in state_dict["state_dict"].items():
            if any([i in k for i in ["model.model.g_ar", "model.model.W_k"]]):
                new_dict[k[6:]] = v
            elif any([i in k for i in ["model.g_ar", "model.W_k", "time_model"]]):
                continue
            elif "model." in k:
                new_dict[k[6:]] = v
            else:
                new_dict[k] = v

        if any(["g_ar" in k for k in new_dict.keys()]):
            model.g_ar = nn.GRU(
                input_size=latent_dim,
                hidden_size=2400,  # default number in infoNCE for langauge
                num_layers=3,
                batch_first=True,
            )
            model.W_k = nn.Linear(2400, latent_dim)
        elif any(["time_model" in k for k in state_dict["state_dict"].keys()]):
            model.fc_mu = nn.Linear(latent_dim, latent_dim)
            model.fc_var = nn.Linear(latent_dim, latent_dim)

        model.load_state_dict(new_dict)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

    def get_checkpoint(
        dataset_name,
        latent_dim,
        base_model="gpt2",
        sec_id=False,
        token_size=None,
        filepath=None,
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_cl_model(
            filepath,
            latent_dim,
            base_model,
            use_section_ids=sec_id,
            token_size=token_size,
        )
        model.to(device)
        model = model.eval()
        return model

    CL_MODEL = get_checkpoint(
        dataset_name="recipe",
        latent_dim=32,
        sec_id=True,
        token_size=len(tokenizer),
        base_model="gpt2",
        filepath="./feature_extractor.pt",
    )
    # CL_MODEL.to(args.device)
else:
    trainer.fit(sys)
    sys.save(directory="./")

# https://github.com/rosewang2008/language_modeling_via_stochastic_processes/blob/main/language_modeling_via_stochastic_processes/transformers/examples/pytorch/text-generation/run_decoding_from_embeddings.py
tc_model = GPT2TimeLMHeadModel.from_pretrained("gpt2")
tc_model._config.use_contrastive_embeddings = True
decoder6 = TimeControl(tc_model, sys.model, random_seed=22)
tc_output = decoder6.generate_text(
    prompt=content,
    max_length=20,
)
txtop6 = tokenizer.decode(tc_output[0], skip_special_tokens=True)
print(txtop6)
