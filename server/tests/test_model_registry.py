import os
from providers.model_registry import choose_model_for_role


def test_choose_model_known_role():
    model = choose_model_for_role("thinkers")
    assert isinstance(model, str)
    assert model in ["Nemotron-4-340B-Instruct", "Llama-3.1-405B-Instruct-NIM", "Mistral-Large-NIM", "Nemotron-4-15B-Instruct-NIM", "nvidia/llama-3.1-8b-instruct"]


def test_choose_model_unknown_role():
    model = choose_model_for_role("unknown_role_xyz")
    assert model == "nvidia/llama-3.1-8b-instruct"
