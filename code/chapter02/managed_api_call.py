"""Listing 2.1: calling a managed API (Anthropic Python SDK).

The "buy" side of chapter 2's buy-versus-build decision in one call. Install the SDK
(`pip install anthropic`, also part of the `chapter03` extra) and export ANTHROPIC_API_KEY,
then run from the code/ directory:

    python -m chapter02.managed_api_call

The same shape works for OpenAI, Gemini, and Bedrock; only the client object, the key
variable, and the model= value change.
"""

import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Explain the difference between a stock and a bond "
                       "in two sentences.",
        }
    ],
)
print(message.content[0].text)
