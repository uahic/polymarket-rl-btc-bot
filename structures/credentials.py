import json
from dataclasses import dataclass


@dataclass
class ClobCredentials:
    """User-level API credentials for CLOB."""

    api_key: str
    secret: str
    passphrase: str

    @classmethod
    def load(cls, filepath: str) -> "ClobCredentials":
        """Load credentials from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            api_key=data.get("apiKey", ""),
            secret=data.get("secret", ""),
            passphrase=data.get("passphrase", ""),
        )

    def is_valid(self) -> bool:
        """Check if credentials are valid."""
        return bool(self.api_key and self.secret and self.passphrase)
