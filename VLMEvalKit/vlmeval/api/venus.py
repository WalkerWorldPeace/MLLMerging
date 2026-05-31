#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""OpenAI-compatible API wrapper.

Rewritten to use any OpenAI-compatible endpoint instead of an internal proxy.
Credentials are read from the environment by the parent ``OpenAIWrapper`` — no
secrets are committed:

  - ``OPENAI_API_KEY``  : API key for the OpenAI-compatible endpoint.
  - ``OPENAI_API_BASE`` : endpoint base URL (defaults to the official OpenAI API
                          when unset).
"""
from .gpt import OpenAIWrapper


MODEL_DICT = {
    'gpt-4-1106-preview': 'gpt-4-turbo-2024-04-09',
    'gpt-4o-2024-05-13': 'gpt-4o',
    'gpt-4o-mini-2024-07-18': 'gpt-4o-mini',
}


class VenusAPIWrapper(OpenAIWrapper):
    """Thin OpenAI-compatible wrapper (name kept for backward compatibility).

    Set ``OPENAI_API_KEY`` and, optionally, ``OPENAI_API_BASE`` in the
    environment. Endpoint/key resolution is delegated to :class:`OpenAIWrapper`.
    """

    def __init__(self, model: str, **kwargs):
        if model in MODEL_DICT:
            model = MODEL_DICT[model]
        super().__init__(model, **kwargs)
