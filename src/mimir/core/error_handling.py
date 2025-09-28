"""
Centralized Error Handling and Logging for mimir tool

Provides consistent error handling, logging, and exception types
across the entire application with structured logging and telemetry.
"""

import logging
import sys
from enum import Enum
from typing import Optional, Any, Dict
import structlog
import time


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DocFinderError(Exception):
    """Base exception for mimir specific errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        super().__init__(message)
        self.severity = severity


class SecurityError(DocFinderError):
    """Security-related errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.CRITICAL)


class ConfigurationError(DocFinderError):
    """Configuration-related errors."""

    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.ERROR)


class SearchError(DocFinderError):
    """Search operation errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.WARNING):
        super().__init__(message, severity)


class IndexingError(DocFinderError):
    """Document indexing errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        super().__init__(message, severity)


class ErrorHandler:
    """Centralized error handling and logging system with structured logging."""

    def __init__(self, logger_name: str = "doc_finder", verbose: bool = False):
        """Initialize error handler with structured logging configuration."""
        self.logger_name = logger_name
        self.verbose = verbose
        self._setup_structured_logging()
        self.logger = structlog.get_logger(logger_name)

        # Performance tracking for error analysis
        self.error_start_time = time.time()
        self.error_counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}

    def _setup_structured_logging(self):
        """Configure structured logging with processors and formatters."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                self._add_emoji_processor,
                (
                    structlog.dev.ConsoleRenderer(colors=True)
                    if self.verbose
                    else structlog.processors.JSONRenderer()
                ),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure stdlib logging to work with structlog
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.DEBUG if self.verbose else logging.INFO,
        )

    def _add_emoji_processor(self, logger, method_name, event_dict):
        """Add emoji processor for visual feedback."""
        level_emoji = {
            "critical": "🚨",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "debug": "🔍",
        }

        level = event_dict.get("level", "info")
        emoji = level_emoji.get(level, "📝")

        # Add emoji to the event for better visual feedback
        event_dict["emoji"] = emoji
        return event_dict

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        suggestions: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Handle an error with appropriate logging and actionable suggestions.

        Args:
            error: The exception to handle
            context: Context information about where the error occurred
            suggestions: Optional dict with actionable suggestions

        Returns:
            bool: True if the error was handled gracefully, False if it should stop execution
        """
        # Track error counts for monitoring
        if isinstance(error, DocFinderError):
            severity_str = (
                error.severity.value
                if hasattr(error.severity, "value")
                else str(error.severity)
            )
        else:
            severity_str = self._get_error_severity(error)

        self.error_counts[severity_str] = self.error_counts.get(severity_str, 0) + 1

        if isinstance(error, DocFinderError):
            return self._handle_doc_finder_error(error, context, suggestions)
        else:
            return self._handle_generic_error(error, context, suggestions)

    def _handle_doc_finder_error(
        self,
        error: DocFinderError,
        context: str,
        suggestions: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Handle mimir specific errors with structured logging."""
        base_event = {
            "error_type": type(error).__name__,
            "context": context,
            "severity": error.severity.value,
            "execution_time": time.time() - self.error_start_time,
        }

        if suggestions:
            base_event["suggestions"] = suggestions

        message = str(error)

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message, **base_event)
            return False  # Stop execution
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(message, **base_event)
            return False  # Stop execution
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(message, **base_event)
            return True  # Continue execution
        else:
            self.logger.info(message, **base_event)
            return True  # Continue execution

    def _handle_generic_error(
        self,
        error: Exception,
        context: str,
        suggestions: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Handle generic Python exceptions with structured logging and actionable suggestions."""
        base_event = {
            "error_type": type(error).__name__,
            "context": context,
            "execution_time": time.time() - self.error_start_time,
        }

        message = str(error)

        # Generate default suggestions based on error type
        if not suggestions:
            suggestions = self._generate_error_suggestions(error)

        if suggestions:
            base_event["suggestions"] = suggestions

        # Critical system errors
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            self.logger.critical(f"System error: {message}", **base_event)
            return False

        # Permission and file errors with actionable suggestions
        elif isinstance(error, (PermissionError, FileNotFoundError)):
            self.logger.error(f"File system error: {message}", **base_event)
            return True  # Can continue with other operations

        # Network and API errors
        elif isinstance(error, (ConnectionError, TimeoutError)):
            self.logger.warning(f"Network error: {message}", **base_event)
            return True  # Can continue with fallback

        # Import and dependency errors
        elif isinstance(error, ImportError):
            self.logger.warning(f"Missing dependency: {message}", **base_event)
            return True  # Can continue with reduced functionality

        # Unknown errors
        else:
            self.logger.error(f"Unexpected error: {message}", **base_event)
            return True  # Try to continue

    def _generate_error_suggestions(self, error: Exception) -> Dict[str, str]:
        """Generate actionable suggestions based on error type."""
        suggestions = {}

        if isinstance(error, PermissionError):
            suggestions["immediate"] = "Check file/directory permissions"
            suggestions["command"] = "chmod 644 <file> or chmod 755 <directory>"

        elif isinstance(error, FileNotFoundError):
            suggestions["immediate"] = "Verify the file or directory exists"
            suggestions["command"] = (
                "Run 'doc_finder index' to rebuild the document index"
            )

        elif isinstance(error, ImportError):
            missing_module = (
                str(error).split("'")[1] if "'" in str(error) else "unknown"
            )
            suggestions["immediate"] = f"Install missing dependency: {missing_module}"
            suggestions["command"] = f"pip install {missing_module}"

        elif isinstance(error, ConnectionError):
            suggestions["immediate"] = "Check internet connection"
            suggestions["fallback"] = "Use offline search modes or cached data"

        elif isinstance(error, TimeoutError):
            suggestions["immediate"] = "Increase timeout values in configuration"
            suggestions["alternative"] = (
                "Use faster search modes like 'fast' instead of 'smart'"
            )

        return suggestions

    def _get_error_severity(self, error: Exception) -> str:
        """Get severity level for generic exceptions."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return "critical"
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return "error"
        elif isinstance(error, (PermissionError, FileNotFoundError, ImportError)):
            return "warning"
        else:
            return "error"

    def log_info(self, message: str, emoji: str = "ℹ️", **context):
        """Log an informational message with structured context."""
        self.logger.info(message, emoji=emoji, **context)

    def log_warning(self, message: str, emoji: str = "⚠️", **context):
        """Log a warning message with structured context."""
        self.logger.warning(message, emoji=emoji, **context)

    def log_error(self, message: str, emoji: str = "❌", **context):
        """Log an error message with structured context."""
        self.logger.error(message, emoji=emoji, **context)

    def log_success(self, message: str, emoji: str = "✅", **context):
        """Log a success message with structured context."""
        self.logger.info(message, emoji=emoji, **context)

    def log_debug(self, message: str, **context):
        """Log a debug message with structured context."""
        if self.verbose:
            self.logger.debug(message, emoji="🔍", **context)

    def log_progress(self, message: str, current: int, total: int, **context):
        """Log progress with structured context for real-time feedback."""
        progress_pct = (current / total * 100) if total > 0 else 0
        self.logger.info(
            message,
            emoji="📈",
            current=current,
            total=total,
            progress_percent=round(progress_pct, 1),
            **context,
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered for telemetry."""
        runtime_seconds = time.time() - self.error_start_time
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "runtime_seconds": round(runtime_seconds, 2),
            "error_rate_per_minute": round(
                sum(self.error_counts.values()) / max(runtime_seconds / 60, 0.1), 2
            ),
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler(verbose: bool = False) -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(verbose=verbose)
    return _error_handler


def handle_error(
    error: Exception, context: str = "", suggestions: Optional[Dict[str, str]] = None
) -> bool:
    """Convenience function to handle errors using the global handler."""
    return get_error_handler().handle_error(error, context, suggestions)


def log_info(message: str, emoji: str = "ℹ️", **context):
    """Convenience function to log info using the global handler."""
    get_error_handler().log_info(message, emoji, **context)


def log_warning(message: str, emoji: str = "⚠️", **context):
    """Convenience function to log warning using the global handler."""
    get_error_handler().log_warning(message, emoji, **context)


def log_error(message: str, emoji: str = "❌", **context):
    """Convenience function to log error using the global handler."""
    get_error_handler().log_error(message, emoji, **context)


def log_success(message: str, emoji: str = "✅", **context):
    """Convenience function to log success using the global handler."""
    get_error_handler().log_success(message, emoji, **context)


def log_progress(message: str, current: int, total: int, **context):
    """Convenience function to log progress using the global handler."""
    get_error_handler().log_progress(message, current, total, **context)


def get_error_summary() -> Dict[str, Any]:
    """Convenience function to get error summary using the global handler."""
    return get_error_handler().get_error_summary()
