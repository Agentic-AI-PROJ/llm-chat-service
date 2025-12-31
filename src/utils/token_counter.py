def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Approximate token count for text.
    Uses simple heuristic: ~4 characters per token for most models.
    This is a reasonable approximation for tracking purposes.
    """
    if not text:
        return 0
    # Average estimation: 1 token ≈ 4 characters
    # More accurate for English text, slightly less for other languages
    return max(1, len(text) // 4)

def count_messages_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """
    Approximate token count for a list of messages.
    Includes overhead for message formatting.
    """
    total = 0
    for message in messages:
        # Count content tokens
        if "content" in message and message["content"]:
            total += count_tokens(message["content"], model)
        # Add overhead for message formatting (role, etc.)
        # OpenAI models use ~4 tokens per message overhead
        total += 4
    # Add 2 tokens for priming the response
    total += 2
    return total
