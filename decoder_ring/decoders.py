import logging
from random import randint
from typing import Optional, Type, Union

from timecontrol.decode import GPT2TimeLMHeadModel
from torch import Tensor, LongTensor
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, set_seed

GenerationOutput = LongTensor


class GenericDecoder:
    def __init__(self, model: GenerationMixin) -> None:
        self.model = model

    def generate_text(
        self,
        prompt: Optional[Tensor],
        max_length: int = 100,
    ) -> GenerationOutput:
        return self.model.generate(
            prompt,
            do_sample=False,
            max_length=max_length,
        )


class GreedyDecoder(GenericDecoder):
    def __init__(self, model: GenerationMixin) -> None:
        super().__init__(model)


class BeamSearch(GenericDecoder):
    def __init__(
        self,
        model: GenerationMixin,
        early_stopping: Optional[bool] = False,
        length_penalty: Optional[float] = 1.0,
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = 1,
    ) -> None:
        super().__init__(model)
        self.early_stopping = early_stopping or False
        self.length_penalty = length_penalty

        if num_beams is not None and num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
        self.num_beams = num_beams

        if num_beam_groups is not None and num_beam_groups < 1:
            raise ValueError("num_beam_groups must be a positive integer")
        self.num_beam_groups = num_beam_groups

    def set_early_stopping(self, early_stopping: bool) -> None:
        self.early_stopping = self.early_stopping

    def set_length_penalty(self, length_penalty: float) -> None:
        # values < 0.0 make shorter sequences; > 0.0 make longer sequences; default 1.0
        self.length_penalty = length_penalty

    def set_num_beams(self, num_beams: int) -> None:
        if num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
        self.num_beams = num_beams

    def set_num_beam_groups(self, num_beam_groups: int) -> None:
        if num_beam_groups < 1:
            raise ValueError("num_beam_groups must be a positive integer")
        self.num_beams = num_beam_groups

    def validate_params(self) -> None:
        if self.num_beams is None:
            raise ValueError(
                "num_beams must be set in decoder's constructor or set_num_beams before generating text"
            )
        elif self.num_beams == 1:
            logging.warn(
                "One beam (as set in num_beams) is the same as greedy decoder (or random sampling, if sampling is enabled)."
            )

    def generate_text(
        self,
        prompt: Optional[Tensor],
        max_length: int = 100,
    ) -> GenerationOutput:
        self.validate_params()
        return self.model.generate(
            prompt,
            early_stopping=self.early_stopping,
            length_penalty=self.length_penalty,
            num_beams=self.num_beams,
            max_length=max_length,
        )


class BeamSearchWithSampling(BeamSearch):
    def __init__(
        self,
        model: GenerationMixin,
        early_stopping: Optional[bool] = False,
        length_penalty: Optional[float] = 1.0,
        num_beams: Optional[int] = None,
        num_beam_groups: Optional[int] = 1,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
        )
        self.seed = random_seed

    def set_random_seed(self, random_seed: int) -> None:
        self.seed = random_seed

    def generate_text(
        self,
        prompt: Optional[Tensor],
        max_length: int = 100,
    ) -> GenerationOutput:
        self.validate_params()
        if self.seed is None:
            logging.warning(
                "Initalize RandomSampling with a random_seed, or call set_random_seed, for reproducible results."
            )
        else:
            set_seed(self.seed)
        return self.model.generate(
            prompt,
            do_sample=True,
            early_stopping=self.early_stopping,
            length_penalty=self.length_penalty,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
            max_length=max_length,
        )


class RandomSampling(GenericDecoder):
    def __init__(
        self, model: GenerationMixin, random_seed: Optional[int] = None
    ) -> None:
        super().__init__(model)
        self.seed = random_seed

    def set_random_seed(self, random_seed: int) -> None:
        self.seed = random_seed

    def validate_params(self) -> None:
        if self.seed is None:
            logging.warning(
                "Initalize decoder with a random_seed, or call set_random_seed, for reproducible results."
            )
        else:
            set_seed(self.seed)

    def generate_text(
        self,
        prompt: Optional[Tensor],
        max_length: int = 100,
    ) -> GenerationOutput:
        self.validate_params()
        return self.model.generate(prompt, do_sample=True, max_length=max_length)


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
        max_length: int = 100,
    ) -> GenerationOutput:
        if self.typical_p is None:
            raise ValueError(
                "typical_p must be set in TypicalDecoder's constructor or set_typical_p before generating text."
            )
        self.validate_params()
        return self.model.generate(
            prompt,
            do_sample=True,
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
        self.validate_params()
        return self.model.generate(
            prompt,
            do_sample=True,
            max_length=max_length,
            penalty_alpha=self.penalty_alpha,
            top_k=self.top_k,
        )


class TimeControl(RandomSampling):
    def __init__(
        self,
        model: GPT2TimeLMHeadModel,
        trained_encoder: Module,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(model, random_seed=random_seed)
        self.encoder = trained_encoder

    def generate_text(
        self,
        prompt: Optional[Tensor],
        max_length: int = 100,
    ) -> GenerationOutput:
        self.validate_params()
        return self.model.generate(
            prompt,
            do_sample=True,
            max_length=max_length,
        )


AnyDecoderMagicClass = Union[
    Type[BeamSearch],
    Type[BeamSearchWithSampling],
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
            or isinstance(self.decoder, BeamSearchWithSampling)
        ):
            self.decoder.set_random_seed(randint(0, 10_000))
        if isinstance(self.decoder, TypicalDecoder):
            self.decoder.set_typical_p(0.8)
        if isinstance(self.decoder, ContrastiveSearch):
            self.decoder.set_penalty_alpha(0.6)
            self.decoder.set_top_k(4)
        if isinstance(self.decoder, BeamSearch) or isinstance(
            self.decoder, BeamSearchWithSampling
        ):
            self.decoder.set_num_beams(3)

    def write_text(self, prompt: str = "", **kwargs) -> str:
        content = self.tokenizer.encode(prompt, return_tensors="pt")
        tsr = self.decoder.generate_text(prompt=content, **kwargs)
        txt = self.tokenizer.decode(tsr[0], skip_special_tokens=True)
        return txt
