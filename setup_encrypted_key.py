#!/usr/bin/env python3
"""
Setup encrypted private key storage

This script helps you securely store your private key encrypted with a password.
"""
import os
import sys
import getpass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from security.key_store import EncryptedKeyStore

print("=" * 70)
print("Encrypted Private Key Setup")
print("=" * 70)
print("\nThis will encrypt your private key with a password.")
print("You'll need the password each time you run the bot.\n")

# Get private key from .env
private_key = os.environ.get("POLY_PRIVATE_KEY")
if not private_key:
    print("❌ Error: POLY_PRIVATE_KEY not found in .env")
    sys.exit(1)

print(f"✓ Found private key in .env")

# Create credentials directory
creds_dir = Path("credentials")
creds_dir.mkdir(exist_ok=True)

output_file = creds_dir / "private_key.enc"

if output_file.exists():
    print(f"\n⚠️  Warning: {output_file} already exists!")
    response = input("Overwrite? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        sys.exit(0)

# Get password
print("\n🔐 Enter a strong password to encrypt your private key:")
password = getpass.getpass("Password: ")
password_confirm = getpass.getpass("Confirm password: ")

if password != password_confirm:
    print("❌ Passwords don't match!")
    sys.exit(1)

if len(password) < 8:
    print("❌ Password too short! Use at least 8 characters.")
    sys.exit(1)

# Encrypt and save
try:
    manager = EncryptedKeyStore()
    manager.encrypt_and_save(private_key, password, str(output_file))
    print(f"\n✅ Private key encrypted and saved to: {output_file}")
except Exception as e:
    print(f"❌ Failed to encrypt key: {e}")
    sys.exit(1)

# Create secure .env template
secure_env = """# Polymarket Trading Bot - Secure Configuration
# =============================================================================
# Required: Wallet Configuration (ENCRYPTED)
# =============================================================================

# Path to encrypted private key
POLY_ENCRYPTED_KEY_PATH=credentials/private_key.enc

# Your wallet address (safe to store in plain text)
POLY_SAFE_ADDRESS={safe_address}

# =============================================================================
# Required for Gasless Trading: Builder Program Credentials
# =============================================================================

POLY_BUILDER_API_KEY={builder_key}
POLY_BUILDER_API_SECRET={builder_secret}
POLY_BUILDER_API_PASSPHRASE={builder_pass}

# =============================================================================
# Optional: Network Configuration
# =============================================================================

# POLY_RPC_URL=https://polygon-rpc.com
# POLY_CHAIN_ID=137
# POLY_CLOB_HOST=https://clob.polymarket.com

# =============================================================================
# Optional: Trading Defaults
# =============================================================================

POLY_DEFAULT_SIZE=1.0

# =============================================================================
# Optional: Paths and Logging
# =============================================================================

# POLY_DATA_DIR=credentials
POLY_LOG_LEVEL=DEBUG
""".format(
    safe_address=os.environ.get("POLY_SAFE_ADDRESS", ""),
    builder_key=os.environ.get("POLY_BUILDER_API_KEY", ""),
    builder_secret=os.environ.get("POLY_BUILDER_API_SECRET", ""),
    builder_pass=os.environ.get("POLY_BUILDER_API_PASSPHRASE", "")
)

# Save backup of secure config
secure_env_file = ".env.secure"
with open(secure_env_file, "w") as f:
    f.write(secure_env)

print(f"\n📝 Created secure config template: {secure_env_file}")
print("\nNext steps:")
print("1. Backup your current .env: mv .env .env.backup")
print("2. Use the secure version: mv .env.secure .env")
print("3. Update your bot usage to provide password:")
print("\n   Example:")
print("   ```python")
print("   bot = TradingBot(")
print("       encrypted_key_path='credentials/private_key.enc',")
print("       password='your-password',  # Prompt at runtime")
print("       safe_address='...'")
print("   )")
print("   ```")
print("\n⚠️  IMPORTANT: Do NOT delete .env.backup until you verify the encrypted setup works!")
