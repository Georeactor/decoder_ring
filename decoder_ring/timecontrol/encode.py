# Code copied from https://github.com/rosewang2008/language_modeling_via_stochastic_processes
# Language modeling via stochastic processes (ICLR Oral 2022)
# https://arxiv.org/abs/2203.11370
"""
@misc{https://doi.org/10.48550/arxiv.2203.11370,
  doi = {10.48550/ARXIV.2203.11370},
  url = {https://arxiv.org/abs/2203.11370},
  author = {Wang, Rose E and Durmus, Esin and Goodman, Noah and Hashimoto, Tatsunori},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Language modeling via stochastic processes},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

import os, pickle, random

from tqdm import tqdm

import datasets
import pytorch_lightning as pl
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Tokenizer, BertTokenizer

# torch.backends.cudnn.benchmark = True


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False


def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config["optim_params"]["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        # num_workers=config.experiment_params.data_loader_workers,
    )
    return loader


class GPT2OUEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, model_name, finetune_gpt2=False):
        super(GPT2OUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune_gpt2
        self.model_name = model_name
        self._init_model()

    def _init_model(self):
        self.model = GPT2Model.from_pretrained(self.model_name)
        self.model = self.model.eval()
        # turn off all the gradients
        for param in self.model.parameters():
            param.requires_grad = self.finetune
        self.mlp = nn.Linear(self.model.wte.embedding_dim, self.hidden_dim)
        self.feature_extractor = (
            self.create_feature_extractor()
        )  # data_dim -> hidden_dim
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)

        ## NEW AUG 19, turn off bias training.
        self.mlp.apply(weights_init)
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(
            *[
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.latent_dim),
            ]
        )

    def create_log_q(self):
        return nn.Sequential(
            *[
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, 1),
            ]
        )

    def get_gpt2_embeddings(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        return gpt_emb

    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, gpt_emb):
        z = self.mlp(gpt_emb)  # 32, 100
        z = self.feature_extractor(z)
        return z

    def forward(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)


class RecipeNLGData(data.Dataset):
    """WikiSection data"""

    def __init__(
        self,
        train,
        all_dataset,
        config,
        tokenizer_name="GPT2",
        filepath=None,
        seed=1,
    ):
        """ """
        super().__init__()
        self.train = train
        self.all_dataset = all_dataset
        self.config = config

        if self.train:
            self.start_idx, self.end_idx = 0, 1_000
        else:
            self.start_idx, self.end_idx = 500_000, 500_100
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self._set_tokenizer()

        self._process_data()
        print("Done loading dataset.")

        print("Example: ", self.processed_data[0]["sentence"])
        print("Example: ", self.processed_data[10]["sentence"])

    def _process_data(self):
        self.processed_data = []
        for doc_id in tqdm(range(self.start_idx, self.end_idx)):
            doc = self.all_dataset[doc_id]
            doc_info = []
            sentence_counter = 0
            # Put all the document sentences together.
            title = [self.section_ids[0] + " " + doc["title"] + " . "]
            ingredients = [
                self.section_ids[1] + " " + (", ".join(doc["ner"]) + " . ").capitalize()
            ]
            directions = [d[:-1] + " . " for d in doc["directions"]]
            directions[0] = self.section_ids[2] + " " + directions[0]
            gpt2_text = title + ingredients + directions
            gpt2_text = [s for s in gpt2_text if s]
            all_sentences = gpt2_text
            # gpt2_text = "".join(gpt2_text)
            # all_sentences = title + ingredients + directions
            if not all(
                [len(self.tokenizer(s)["input_ids"]) < 1024 for s in all_sentences]
            ):
                continue
            for sentence in all_sentences:
                if not sentence:
                    continue
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_counter,
                    "doc_id": doc_id,
                }
                doc_info.append(sentence_info)
                sentence_counter += 1

            # Track total number of sentences in a document
            for info in doc_info:
                info["total_doc_sentences"] = sentence_counter

            self.processed_data += doc_info

    def _set_tokenizer(self):
        if self.tokenizer_name == "GPT2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token_id
            self.max_length = 1024
        # elif self.tokenizer_name == "BERT":
        #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #     self.max_length = 512
        else:
            raise ValueError("Dont recognize name {}".format(self.tokenizer_name))

        self.section_ids = ["[ TITLE ]", "[ INGREDIENTS ]", "[ DIRECTIONS ]"]
        self.section_names = self.section_ids
        self.cl_eos_str = " . "
        self.tokenizer.add_tokens(self.section_ids + [self.cl_eos_str])
        self.special_tokens = [
            _[0] for _ in self.tokenizer(self.section_ids)["input_ids"]
        ]
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)["input_ids"][0]
        print("CL EOS ID", self.cl_eos_id)

    def tokenize_caption(self, caption, device):
        if self.tokenizer_name == "GPT2":
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors="pt",
            )
            input_ids = output["input_ids"].squeeze(0)
            attention_mask = output["attention_mask"].squeeze(0)
            eos_input_ids = torch.tensor([[self.end_token] * input_ids.shape[0]])
            eos_attention = torch.tensor([[0] * input_ids.shape[0]])
            input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
            attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        elif self.tokenizer_name == "BERT":
            # Prepend [CLS] so I can use the first embedding
            output = self.tokenizer(
                caption,
                padding=True,
                return_tensors="pt",
            )
            input_ids = output["input_ids"].squeeze(0)
            attention_mask = output["attention_mask"].squeeze(0)

        return input_ids.to(device), attention_mask.to(device)

    def __len__(self):
        return len(self.processed_data) - 1


class RecipeDiscourse(RecipeNLGData):
    def __init__(
        self,
        train,
        all_dataset,
        config,
        tokenizer_name="GPT2",
        seed=1,
    ):
        """ """
        super(RecipeDiscourse, self).__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )

    def __getitem__(self, index):
        label = random.randint(0, 1)  # either in- or out-of-order

        if label:  # in-order
            if (
                self.processed_data[index]["doc_id"]
                != self.processed_data[index + 1]["doc_id"]
            ):
                index -= 1
            y_t = self.processed_data[index]["sentence"]
            y_tp1 = self.processed_data[index + 1]["sentence"]
        else:
            y_t = self.processed_data[index]["sentence"]
            random_idx = random.randint(
                0, len(self.processed_data) - 1
            )  # either in- or out-of-order
            y_tp1 = self.processed_data[random_idx]["sentence"]

        if self.one_hot_labels:
            labels = torch.zeros(2)
            labels[label] = 1.0
            label = labels

        result = {"y_t": y_t, "y_tp1": y_tp1, "label": label, "idx": index}
        return result


class RecipeTriplet(RecipeNLGData):
    def __init__(
        self,
        train,
        all_dataset,
        config,
        tokenizer_name="GPT2",
        seed=1,
    ):
        """ """
        super(RecipeTriplet, self).__init__(
            train=train,
            all_dataset=all_dataset,
            config=config,
            tokenizer_name=tokenizer_name,
            seed=seed,
        )

    def __getitem__(self, index):
        utterance = self.processed_data[index]
        sentence_num = utterance["sentence_id"]

        # Check if index is start of a seq. If so -> +2
        if sentence_num == 0:
            index += 2
        if sentence_num == 1:
            index += 1

        # Update
        utterance = self.processed_data[index]
        sentence_num = utterance["sentence_id"]

        # TRIAL 2: Sample all random points, t, t', t''
        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]["sentence"]
        y_t = self.processed_data[index - T + t2]["sentence"]
        y_T = self.processed_data[index]["sentence"]

        t_ = t1
        t = t2

        total_doc = utterance["total_doc_sentences"]
        result = {
            "y_0": y_0,
            "y_t": y_t,
            "y_T": y_T,
            "t_": t_,
            "t": t,
            "T": T,
            "total_t": total_doc,
        }
        return result


NAME2DATASET = {
    # 'wikisection': wikisection.WikisectionTPK,
    "recipe": RecipeTriplet,
    # 'wikihow': wikihow.WikihowTPK,
    # 'roc_stories': roc_stories.ROCStoriesTPK,
    # 'tm2': tm2.TM2TPK,
    # 'tickettalk': tickettalk.TicketTalkTPK,
}


class BrownianBridgeLoss(object):
    """Everything is a brownian bridge...
    p(z_t | mu_0, mu_T) = \mathcal{N}(mu_0 * t/T + mu_T * (1-t/T), I t*(T-t)/T)
    normalization constant: -1/(2 * t*(T-t)/T)
    """

    def __init__(
        self,
        z_0,
        z_t,
        z_T,
        t_,
        t,
        T,
        alpha,
        var,
        log_q_y_T,
        loss_type,
        eps,
        max_seq_len,
        C_eta=None,
        label=None,
    ):
        super().__init__()
        self.log_q_y_T = log_q_y_T
        self.z_0 = z_0
        self.z_t = z_t
        self.z_T = z_T
        self.t_ = t_
        self.t = t
        self.T = T
        self.alpha = alpha
        self.var = var
        NAME2LOSS = {
            "simclr": self.simclr_loss,
        }
        self.loss_f = NAME2LOSS[loss_type]
        self.eps = eps
        self.max_seq_len = max_seq_len
        self.sigmoid = nn.Sigmoid()
        self.label = label

        if C_eta is None:
            C_eta = 0.0
        self.C_eta = C_eta
        self.end_pin_val = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _log_p(self, z_0, z_t, z_T, t_0, t_1, t_2):
        T = t_2 - t_0
        t = t_1 - t_0

        alpha = (t / (T + self.eps)).view(-1, 1)
        delta = z_0 * (1 - alpha) + z_T * (alpha) - z_t
        var = t * (T - t) / (T + self.eps)
        log_p = (
            -1 / (2 * var + self.eps) * (delta * delta).sum(-1) + self.C_eta
        )  # (512,)
        if len(log_p.shape) > 1:  # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, z_0, z_T, z_t, t_, t, T):
        """
        Calculating log p(z_tp1, z_t) = -|| h(z_{t+dt}) - h(z_t)(1-dt)||^2_2
        """
        log_p = self._log_p(z_0=z_0, z_t=z_t, z_T=z_T, t_0=t_, t_1=t, t_2=T)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_T
        logit = log_p  # - log_q
        return logit  # should be (bsz, 1)

    def reg_loss(self):
        loss = 0.0
        mse_loss_f = nn.MSELoss()
        # start reg
        start_idxs = torch.where((self.t_) == 0)[0]
        if start_idxs.nelement():
            vals = self.z_0[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        # end reg
        end_idxs = torch.where((self.T) == self.max_seq_len - 1)[0]
        if end_idxs.nelement():
            vals = torch.abs(self.z_T[end_idxs, :])
            end_reg = mse_loss_f(
                vals, torch.ones(vals.shape, device=self.device) * self.end_pin_val
            )
            loss += end_reg
        return loss

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}
        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(
            z_0=self.z_0, z_T=self.z_T, z_t=self.z_t, t_=self.t_, t=self.t, T=self.T
        )
        pos_probs = torch.exp(pos_logit)  # (bsz,1)
        for idx in range(self.z_T.shape[0]):
            # Negative pair: logits over all possible contrasts
            # Nominal contrast for random triplet - contrast from in between
            neg_i_logit = self._logit(
                z_0=self.z_0,
                z_T=self.z_T,
                z_t=self.z_t[idx],
                t_=self.t_,
                t=self.t[idx],
                T=self.T,
            )
            neg_i_probs = torch.exp(neg_i_logit)  # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.z_T.shape[0]
        # Regularization for pinning start and end of bridge
        reg_loss = self.reg_loss()
        loss += reg_loss
        return loss

    def get_loss(self):
        return self.loss_f()


class BrownianBridgeSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config["model_params"]["name"]
        self._set_dataset()
        self._set_language_encoder()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config["optim_params"]["learning_rate"],
            momentum=self.config["optim_params"]["momentum"],
        )
        return [optimizer], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def _set_dataset(self):
        dname = self.config["data_params"]["name"]
        if "recipe" == dname:
            self.data_dir = self.config["data_params"][
                "path"
            ]  # constants.PATH2RECIPENLG
            self.all_dataset = datasets.load_dataset(
                "recipe_nlg", data_dir=self.data_dir
            )["train"]
        elif "wikihow" == dname:
            self.data_name = constants.PATH2WIKIHOW
            with open(self.data_name, "rb") as f:
                self.all_dataset = pickle.load(f)
        else:
            self.all_dataset = None

        dataset = NAME2DATASET[dname]
        self.train_dataset = dataset(
            train=True,
            # seed=self.config['data_params']['data_seed'],
            all_dataset=self.all_dataset,
            config=self.config,
        )
        self.test_dataset = dataset(
            train=False,
            # seed=self.config['data_params']['data_seed'],
            all_dataset=self.all_dataset,
            config=self.config,
        )

    def set_to_train(self):
        pass

    def _set_language_encoder(self):
        self.model = GPT2OUEncoder(
            hidden_dim=self.config["model_params"]["hidden_size"],
            latent_dim=self.config["model_params"]["latent_dim"],
            model_name=self.model_name,
            finetune_gpt2=False,
        )

        self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        for p in self.model.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_feats(self, obs):
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs, device=self.device
        )
        input_ids_i = input_ids_i[:, : self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, : self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        if "y_0" in batch:
            obs_0 = batch["y_0"]
        else:
            obs_0 = ""
        obs_t = batch["y_t"]
        obs_T = batch["y_T"]
        t_s = batch["t_"].float()
        ts = batch["t"].float()
        Ts = batch["T"].float()
        feats_0 = self.get_feats(obs_0)
        feats_t = self.get_feats(obs_t)
        feats_T = self.get_feats(obs_T)
        log_q_y_tp1 = self.model.get_log_q(feats_t)
        loss_fn = BrownianBridgeLoss(
            z_0=feats_0,
            z_t=feats_t,
            z_T=feats_T,
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            log_q_y_T=log_q_y_tp1,
            loss_type=self.config["loss_params"]["name"],
            eps=self.config["model_params"]["eps"],
            max_seq_len=batch["total_t"].float(),
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch, i):
        loss = self.get_losses_for_batch(batch=batch, batch_idx=i)
        self.log("test_loss", loss.cpu().detach().numpy(), prog_bar=True, on_step=True)
        return loss
