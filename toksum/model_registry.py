"""
Custom exceptions for the token counting module.
"""

class UnsupportedModelError(ValueError):
    """Exception raised when a model is not supported."""
    
    def __init__(self, model: str, supported_models: list = None):
        self.model = model
        self.supported_models = supported_models or []
        message = f"Model '{model}' is not supported."
        if supported_models:
            message += f" Supported models: {', '.join(sorted(supported_models)[:10])}{'...' if len(supported_models) > 10 else ''}"
        super().__init__(message)


class TokenizationError(Exception):
    """Exception raised when tokenization fails."""
    
    def __init__(self, message: str, model: str = None, text_preview: str = None):
        self.model = model
        self.text_preview = text_preview
        full_message = message
        if model:
            full_message += f" (model: {model})"
        if text_preview:
            preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
            full_message += f" (text: '{preview}')"
        super().__init__(full_message)
