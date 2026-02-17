"""
Quick script to check what type of address is shown on Polymarket
"""
import logging
from web3 import Web3

logger = logging.getLogger(__name__)

# The address shown in Polymarket settings
polymarket_address = "0xF7F5C8669891e4c62dF672864E78a0B4B39757f8"

# The address from your config
config_address = "0x0177863ebd6b36b4cd0f667752fd825fd493e7ed"

# Connect to Polygon
rpc_url = "https://polygon-rpc.com"
w3 = Web3(Web3.HTTPProvider(rpc_url))

logger.info("="*70)
logger.info("ADDRESS ANALYSIS")
logger.info("="*70)

def check_address(addr, label):
    logger.info(f"\n{label}")
    logger.info(f"Address: {addr}")

    checksum = Web3.to_checksum_address(addr)

    # Check if it's a contract
    code = w3.eth.get_code(checksum)

    if code == b'' or code == b'0x':
        logger.info(f"Type: EOA (Externally Owned Account)")
        logger.info(f"This is a regular wallet address")
    else:
        logger.info(f"Type: CONTRACT")
        logger.info(f"Bytecode size: {len(code)} bytes")

        # Try to determine contract type
        try:
            # Check for Gnosis Safe's getOwners() function
            owners_sig = Web3.keccak(text="getOwners()")[:4].hex()
            result = w3.eth.call({'to': checksum, 'data': owners_sig})

            from eth_abi import decode
            owners = decode(['address[]'], result)[0]

            logger.info(f"\n✅ This is a GNOSIS SAFE")
            logger.info(f"Number of owners: {len(owners)}")
            logger.info(f"\nSafe Owners:")
            for i, owner in enumerate(owners, 1):
                logger.info(f"  {i}. {owner}")

        except:
            logger.info(f"\nℹ️  Might be a Polymarket Proxy or other contract")

    logger.info("-"*70)

# Check both addresses
check_address(polymarket_address, "📍 ADDRESS FROM POLYMARKET SETTINGS")
check_address(config_address, "📍 ADDRESS FROM YOUR CONFIG")

logger.info("\n" + "="*70)
logger.info("RECOMMENDATION")
logger.info("="*70)
logger.info("\n1. The address shown in Polymarket settings is your TRADING address")
logger.info("2. This is the address that should be in config.yaml as 'safe_address'")
logger.info("3. Your private key should control this address (or be an owner of it)")
logger.info("\nℹ️  Use the address from Polymarket settings: 0xF7F5C8669891e4c62dF672864E78a0B4B39757f8")
logger.info("="*70)
