import logging
from typing import Optional, Union

from torch import Tensor, LongTensor
from transformers import set_seed
from transformers.generation_utils import (
    GenerationMixin,
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
)

GenerationOutput = Union[
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
    LongTensor,
]


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
        super().__init__(model)
        self.seed = random_seed
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
                "Initalize RandomSampling with a random_seed, or call set_random_seed, for reproducible results."
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
