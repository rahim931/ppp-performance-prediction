from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class LLMResponse:
    """
    Standardized response object returned by the LLM client.

    This object guarantees a consistent interface for all experiments,
    independent of the underlying API or model.
    """
    text: str                     # Main generated text from the LLM
    raw: Dict[str, Any]           # Raw API response (for debugging / logging)
    model: str                    # Model name used for generation
    prompt_hash: str              # Hash of the prompt (for caching / reproducibility)
    latency_s: float              # Request latency in seconds
    cached: bool                  # Whether the response was returned from cache


class LLMClient:
    """
    Abstract LLM client interface.

    Concrete implementations (e.g. IMI API client) must implement
    the generate() method.
    """

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Send a prompt to the LLM and return a standardized response.

        Parameters
        ----------
        prompt : str
            The full prompt sent to the language model.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum number of tokens to generate.
        stop : Optional[List[str]]
            Optional stop sequences.
        meta : Optional[Dict[str, Any]]
            Optional metadata (e.g. dataset name, heuristic id).

        Returns
        -------
        LLMResponse
            Standardized LLM response object.
        """
        raise NotImplementedError(
            "LLMClient.generate() must be implemented by a concrete client."
        )
