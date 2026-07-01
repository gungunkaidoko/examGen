"""
LLM Client
----------
Wrapper around LangChain + AWS Bedrock (Claude Sonnet 4.5).
Provides two clients:
  - generator_llm  : used for question generation (higher temperature for diversity)
  - validator_llm  : used for correctness checks (temperature=0 for determinism)
"""

import os
import boto3
from langchain_aws import ChatBedrock
from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    BEDROCK_INFERENCE_PROFILE_ARN,
    MODEL_ID,
)


def _build_llm(temperature: float, max_tokens: int = 8192) -> ChatBedrock:
    """Factory for Claude via AWS Bedrock LLM instances."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise EnvironmentError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are not set. "
            "Check your .env file."
        )

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,
    )
    bedrock_client = session.client("bedrock-runtime")

    # Use inference profile ARN if available, otherwise fall back to model ID
    model = BEDROCK_INFERENCE_PROFILE_ARN if BEDROCK_INFERENCE_PROFILE_ARN else MODEL_ID

    return ChatBedrock(
        client=bedrock_client,
        model_id=model,
        provider="anthropic",          # required by langchain-aws when model_id is an ARN
        model_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )


# Singleton instances — import these everywhere
generator_llm = _build_llm(temperature=0.9, max_tokens=8192)   # diverse question generation
validator_llm = _build_llm(temperature=0.0, max_tokens=1024)   # deterministic verification
