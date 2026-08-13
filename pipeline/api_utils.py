import time

import cohere
import openai

# transient, worth a retry: rate limits, timeouts, connection issues, the provider's own 5xx.
# anything else (bad API key, bad request, etc.) fails immediately, retrying it 3 times would
# just waste ~7 seconds before failing with the same error anyway.
RETRYABLE_OPENAI_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
RETRYABLE_COHERE_ERRORS = (cohere.TooManyRequestsError, cohere.ServiceUnavailableError, cohere.GatewayTimeoutError, cohere.InternalServerError, cohere.ClientClosedRequestError)


class ExternalAPIError(Exception):
    """Raised when a call to OpenAI, Cohere, or Pinecone fails (after retries for transient
    errors) or comes back with something the pipeline can't use, so callers get one clear,
    user-facing message instead of a raw SDK traceback or a confusing crash further downstream."""


def call_with_retries(fn, *, service: str, retryable_errors: tuple = (Exception,), attempts: int = 3, base_delay: float = 1.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except retryable_errors as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            raise ExternalAPIError(f"{service} call failed: {e}") from e
    raise ExternalAPIError(f"{service} did not respond after {attempts} attempts: {last_error}") from last_error