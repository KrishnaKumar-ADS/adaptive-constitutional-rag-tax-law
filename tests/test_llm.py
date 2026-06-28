import pytest
from unittest.mock import patch, MagicMock
from src.llm.openrouter import inference_groq

@patch("src.llm.openrouter.client")
def test_inference_groq(mock_client):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a mocked answer."
    
    mock_client.chat.completions.create.return_value = mock_response
    
    prompt = "What is the capital of India?"
    answer = inference_groq(prompt)
    
    assert answer == "This is a mocked answer."
    mock_client.chat.completions.create.assert_called_once()
    
    # Check that it uses the prompt correctly
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][0]["content"] == prompt
    assert call_kwargs["messages"][0]["role"] == "user"
