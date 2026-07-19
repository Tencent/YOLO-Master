# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis


class YOLOMasterError(Exception):
    """Base exception for optional YOLO-Master extensions."""

    def __init__(self, message: str = ""):
        super().__init__(emojis(message))


class HUBModelError(YOLOMasterError):
    """Exception raised when a model cannot be found or retrieved from Ultralytics HUB.

    This custom exception is used specifically for handling errors related to model fetching in Ultralytics YOLO. The
    error message is processed to include emojis for better user experience.

    Attributes:
        message (str): The error message displayed when the exception is raised.

    Methods:
        __init__: Initialize the HUBModelError with a custom message.

    Examples:
        >>> try:
        ...     # Code that might fail to find a model
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # Displays the emoji-enhanced error message
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """Initialize a HUBModelError exception.

        This exception is raised when a requested model is not found or cannot be retrieved from Ultralytics HUB. The
        message is processed to include emojis for better user experience.

        Args:
            message (str, optional): The error message to display when the exception is raised.
        """
        super().__init__(emojis(message))


class MoERouterError(YOLOMasterError):
    """Raised when a routed module receives invalid input or configuration."""


class PEFTPlannerError(YOLOMasterError):
    """Base error for optional PEFT planner operational failures."""


class PEFTRefusalError(PEFTPlannerError):
    """Raised when the optional PEFT planner makes an explicit refusal decision."""

    def __init__(self, reason: str = "", predicted_delta: float = 0.0):
        self.reason = reason
        self.predicted_delta = predicted_delta
        message = f"PEFT Planner refused: {reason}" if reason else "PEFT Planner refused."
        if predicted_delta:
            message += f" (predicted ΔmAP={predicted_delta:.3f})"
        super().__init__(message)


class ShapeMismatchError(YOLOMasterError):
    """Raised when a routed tensor violates an expected shape contract."""

    def __init__(self, expected, actual, context: str = ""):
        self.expected = expected
        self.actual = actual
        self.context = context
        message = f"Shape mismatch: expected {expected}, got {actual}"
        if context:
            message += f" [{context}]"
        super().__init__(message)
