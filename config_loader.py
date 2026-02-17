import logging
import os
import sys
from pathlib import Path
from getpass import getpass

sys.path.insert(0, str(Path(__file__).parent))
from logger.colors import Colors
from security.key_store import EncryptedKeyStore, CryptoError, InvalidPasswordError
from config import Config

logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Print a section header."""
    logger.info(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.BLUE}{title:^50}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.RESET}\n")


def print_success(msg: str) -> None:
    logger.info(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    logger.info(f"{Colors.RED}✗{Colors.RESET} {msg}")


def load_config() -> Config:
    """Load configuration from config.yaml."""
    if not os.path.exists("config.yaml"):
        print_error("config.yaml not found!")
        logger.info("\nPlease run the setup first:")
        logger.info(f"  {Colors.CYAN}python scripts/setup.py{Colors.RESET}")
        sys.exit(1)

    try:
        config = Config.load("config.yaml")
        errors = config.validate()

        if errors:
            print_error("Configuration validation failed:")
            for error in errors:
                logger.info(f"  - {error}")
            sys.exit(1)

        return config
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)


def get_private_key_from_env() -> str:
    """Get private key from environment variable."""
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    if not private_key:
        print_error("POLY_PRIVATE_KEY environment variable not set!")
        sys.exit(1)
    return private_key


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    config = Config.from_env()
    errors = config.validate()

    if errors:
        print_error("Configuration validation failed:")
        for error in errors:
            logger.info(f"  - {error}")
        sys.exit(1)

    return config


def decrypt_private_key() -> str:
    """Decrypt private key from encrypted file."""
    key_path = "credentials/encrypted_key.json"

    if not os.path.exists(key_path):
        print_error("Encrypted key not found!")
        logger.info("\nPlease run the setup first:")
        logger.info(f"  {Colors.CYAN}python scripts/setup.py{Colors.RESET}")
        sys.exit(1)

    logger.info(f"{Colors.BOLD}Enter decryption password:{Colors.RESET}")

    while True:
        password = getpass("Password: ")

        try:
            manager = EncryptedKeyStore()
            private_key = manager.load_and_decrypt(password, key_path)
            print_success("Private key decrypted")
            return private_key

        except InvalidPasswordError:
            print_error("Invalid password, try again")
        except CryptoError as e:
            print_error(f"Failed to decrypt: {e}")
            sys.exit(1)
