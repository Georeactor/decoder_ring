import logging
from random import randint
from typing import Optional, Type, Union

from torch import Tensor, LongTensor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, set_seed

GenerationOutput = LongTensor


class GenericDecoder:
    def __init__(self, model: GenerationMixin) -> None:
        self.model = model

    def generate_text(
        self,
        prompt: Optional[Tensor],
        early_stopping: bool = True,
        max_length: int = 100,
    ) -> GenerationOutput:
        return self.model.generate(
            prompt,
            do_sample=False,
            early_stopping=early_stopping,
            max_length=max_length,
        )


class GreedyDecoder(GenericDecoder):
    def __init__(self, model: GenerationMixin) -> None:
        super().__init__(model)


class RandomSampling(GenericDecoder):
    def __init__(
        self, model: GenerationMixin, random_seed: Optional[int] = None
    ) -> None:
        super().__init__(model)
        self.seed = random_seed

    def set_random_seed(self, random_seed: int) -> None:
        if random_seed is not None:
            self.seed = random_seed

    def generate_text(
        self,
        prompt: Optional[Tensor],
        early_stopping: bool = True,
        max_length: int = 100,
    ) -> GenerationOutput:
        if self.seed is None:
            logging.warning(
                "Initalize RandomSampling with a random_seed, or call set_random_seed, for reproducible results."
            )
        else:
            set_seed(self.seed)
        return self.model.generate(
            prompt, do_sample=True, early_stopping=early_stopping, max_length=max_length
        )


class TypicalDecoder(RandomSampling):
    def __init__(
        self,
        model: GenerationMixin,
        random_seed: Optional[int] = None,
        typical_p: Optional[float] = None,
    ) -> None:
        super().__init__(model, random_seed=random_seed)
        if typical_p is not None and (typical_p <= 0 or typical_p >= 1):
            raise ValueError("typical_p must be a number between 0 and 1")
        self.typical_p = typical_p

    def set_typical_p(self, typical_p: float) -> None:
        if typical_p <= 0 or typical_p >= 1:
            raise ValueError("typical_p must be a number between 0 and 1")
        self.typical_p = typical_p

    def generate_text(
        self,
        prompt: Optional[Tensor],
        early_stopping: bool = True,
        max_length: int = 100,
    ) -> GenerationOutput:
        if self.typical_p is None:
            raise ValueError(
                "typical_p must be set in TypicalDecoder's constructor or set_typical_p before generating text."
            )
        if self.seed is None:
            logging.warning(
                "Initalize TypicalDecoder with a random_seed, or call set_random_seed, for reproducible results."
            )
        else:
            set_seed(self.seed)
        return self.model.generate(
            prompt,
            do_sample=True,
            early_stopping=early_stopping,
            max_length=max_length,
            typical_p=self.typical_p,
        )


class ContrastiveSearch(RandomSampling):
    def __init__(
        self,
        model: GenerationMixin,
        random_seed: Optional[int] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__(model, random_seed=random_seed)
        if penalty_alpha is not None and (penalty_alpha <= 0 or penalty_alpha >= 1):
            raise ValueError("penalty_alpha must be a number between 0 and 1")
        self.penalty_alpha = penalty_alpha
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be an integer greater than 0")
        self.top_k = top_k

    def set_penalty_alpha(self, penalty_alpha: float) -> None:
        if penalty_alpha is not None and (penalty_alpha <= 0 or penalty_alpha >= 1):
            raise ValueError("penalty_alpha must be a number between 0 and 1")
        self.penalty_alpha = penalty_alpha

    def set_top_k(self, top_k: int) -> None:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be an integer greater than 0")
        self.top_k = top_k

    def generate_text(
        self,
        prompt: Optional[Tensor],
        early_stopping: bool = True,
        max_length: int = 100,
    ) -> GenerationOutput:
        if self.penalty_alpha is None:
            raise ValueError(
                "penalty_alpha must be set in ContrastiveSearch's constructor or set_penalty_alpha before generating text."
            )
        if self.top_k is None:
            raise ValueError(
                "top_k must be set in ContrastiveSearch's constructor or set_top_k before generating text."
            )
        if self.seed is None:
            logging.warning(
                "Initalize ContrastiveSearch with a random_seed, or call set_random_seed, for reproducible results."
            )
        else:
            set_seed(self.seed)
        return self.model.generate(
            prompt,
            do_sample=True,
            early_stopping=early_stopping,
            max_length=max_length,
            penalty_alpha=self.penalty_alpha,
            top_k=self.top_k,
        )


AnyDecoderMagicClass = Union[
    Type[ContrastiveSearch],
    Type[GreedyDecoder],
    Type[RandomSampling],
    Type[TypicalDecoder],
]


class BasicWriter:
    def __init__(
        self,
        model_name: str,
        decoder: AnyDecoderMagicClass,
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.decoder = decoder(self.model)
        if (
            isinstance(self.decoder, RandomSampling)
            or isinstance(self.decoder, TypicalDecoder)
            or isinstance(self.decoder, ContrastiveSearch)
        ):
            self.decoder.set_random_seed(randint(0, 10_000))
        if isinstance(self.decoder, TypicalDecoder):
            self.decoder.set_typical_p(0.8)
        if isinstance(self.decoder, ContrastiveSearch):
            self.decoder.set_penalty_alpha(0.6)
            self.decoder.set_top_k(4)

    def write_text(self, prompt: str = "", **kwargs) -> str:
        content = self.tokenizer.encode(prompt, return_tensors="pt")
        tsr = self.decoder.generate_text(prompt=content, **kwargs)
        txt = self.tokenizer.decode(tsr[0], skip_special_tokens=True)
        return txt
