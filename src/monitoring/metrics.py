"""Lightweight API runtime metrics for EduPredict."""

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class APIMetrics:
    """Track basic API request and prediction statistics."""

    total_requests: int = 0
    successful_predictions: int = 0
    errors: int = 0
    total_latency_seconds: float = 0.0

    _lock: Lock = field(
        default_factory=Lock,
        repr=False,
    )

    def observe(
        self,
        latency_seconds: float,
        prediction: bool = False,
        error: bool = False,
    ) -> None:
        """Record one API request."""

        with self._lock:
            self.total_requests += 1
            self.total_latency_seconds += latency_seconds

            if prediction:
                self.successful_predictions += 1

            if error:
                self.errors += 1

    def snapshot(self) -> dict:
        """Return current metrics as a JSON-safe dictionary."""

        with self._lock:
            average_latency = (
                self.total_latency_seconds
                / self.total_requests
                if self.total_requests
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "successful_predictions": self.successful_predictions,
                "errors": self.errors,
                "average_latency_seconds": round(
                    average_latency,
                    6,
                ),
            }


metrics = APIMetrics()
