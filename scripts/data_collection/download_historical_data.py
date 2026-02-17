"""
Download historical market data for offline training.

This script downloads data from:
1. Polymarket CLOB API - price history
2. Binance Futures API - klines, aggregated trades
3. Binance Spot API - price history

Data is saved to parquet files for fast loading during training.

Usage:
    python scripts/data_collection/download_historical_data.py \\
        --assets BTC ETH SOL XRP \\
        --start-date 2026-01-15 \\
        --end-date 2026-02-15 \\
        --output-dir data/historical
"""

import argparse
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
BINANCE_FUTURES_API = "https://fapi.binance.com"
BINANCE_SPOT_API = "https://api.binance.com"


def get_token_id_for_market(asset_lower: str, timestamp: int, max_retries: int = 3) -> Optional[str]:
    """
    Get token ID for a specific 15-min market with retry logic.

    Args:
        asset_lower: Asset symbol in lowercase
        timestamp: Unix timestamp for market end time
        max_retries: Maximum number of retry attempts

    Returns:
        Token ID for UP outcome, or None if not found after retries
    """
    slug = f"{asset_lower}-updown-15m-{timestamp}"

    for attempt in range(max_retries):
        try:
            # Query Gamma API for this market
            url = f"{GAMMA_API}/events?slug={slug}"
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                return None

            events = resp.json()
            if not events:
                return None

            event = events[0]

            # Get condition ID
            condition_id = None
            for market in event.get("markets", []):
                condition_id = market.get("conditionId")
                if condition_id:
                    break

            if not condition_id:
                return None

            # Get token IDs from CLOB with retry
            clob_url = f"{POLYMARKET_CLOB_API}/markets/{condition_id}"
            clob_resp = requests.get(clob_url, timeout=10)

            if clob_resp.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

            clob_data = clob_resp.json()
            tokens = clob_data.get("tokens", [])

            # Find UP token
            for token in tokens:
                if token.get("outcome", "").lower() == "up":
                    return token.get("token_id")

            return None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                logger.warning(f"  Timeout/connection error for {slug}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"  ERROR: Failed after {max_retries} attempts for {slug}: {e}")
                return None
        except Exception as e:
            logger.info(f"  ERROR checking {slug}: {e}")
            return None

    return None


def get_historical_15m_markets(
    asset: str,
    start_date: datetime,
    end_date: datetime,
) -> Tuple[int, int, str]:
    """
    Get timestamp range for 15-min markets.

    Since Polymarket 15-min markets switch deterministically every 15 minutes,
    we only need to verify the first and last market exist, then we can
    generate timestamps as needed during download.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP)
        start_date: Start date
        end_date: End date

    Returns:
        Tuple of (start_timestamp, end_timestamp, asset_lower)
        Returns (0, 0, "") if markets don't exist
    """
    logger.info(f"\nFinding 15-min markets for {asset} from {start_date.date()} to {end_date.date()}")

    asset_lower = asset.lower()

    # Round to 15-min boundaries
    current = start_date.replace(tzinfo=timezone.utc)
    end = end_date.replace(tzinfo=timezone.utc)

    start_ts = int(current.timestamp())
    start_ts = (start_ts // 900) * 900
    end_ts = int(end.timestamp())
    end_ts = (end_ts // 900) * 900

    # Calculate total number of markets
    num_markets = (end_ts - start_ts) // 900 + 1
    logger.info(f"  Expected {num_markets} markets between timestamps {start_ts} and {end_ts}")

    # Verify first market exists
    logger.info(f"  Verifying first market at timestamp {start_ts}...")
    first_token_id = get_token_id_for_market(asset_lower, start_ts)

    if not first_token_id:
        logger.error(f"  ERROR: First market not found at timestamp {start_ts}")
        logger.info(f"  Slug would be: {asset_lower}-updown-15m-{start_ts}")
        return (0, 0, "")

    logger.info(f"  ✓ First market found: {first_token_id}")

    # Verify last market exists
    logger.info(f"  Verifying last market at timestamp {end_ts}...")
    last_token_id = get_token_id_for_market(asset_lower, end_ts)

    if not last_token_id:
        logger.error(f"  ERROR: Last market not found at timestamp {end_ts}")
        logger.info(f"  Slug would be: {asset_lower}-updown-15m-{end_ts}")
        return (0, 0, "")

    logger.info(f"  ✓ Last market found: {last_token_id}")
    logger.info(f"  ✓ Timestamp range validated: {num_markets} markets available")

    return (start_ts, end_ts, asset_lower)


def download_polymarket_prices(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1,  # 1 minute resolution
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Download Polymarket price history for a token with retry logic.

    Args:
        token_id: CLOB token ID
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds)
        fidelity: Data resolution in minutes
        max_retries: Maximum number of retry attempts

    Returns:
        DataFrame with columns: timestamp, price
    """

    url = f"{POLYMARKET_CLOB_API}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            # Check for rate limiting (429 status code)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"  Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            # Check if response is an error message
            if isinstance(data, str):
                logger.error(f"  ERROR: API returned string: {data}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])

            # Check if response is an error dict
            if isinstance(data, dict) and "error" in data:
                logger.error(f"  ERROR: API returned error: {data.get('error')}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])

            # Parse response: {"history": [{t: timestamp, p: price}, ...]}
            if isinstance(data, dict) and "history" in data:
                history = data["history"]
            elif isinstance(data, list):
                history = data
            else:
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])

            if not history:
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])

            df = pd.DataFrame([
                {"timestamp": item["t"], "price": item["p"], "token_id": token_id}
                for item in history
            ])

            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Longer waits to avoid rate limits
                logger.warning(f"  Timeout/connection error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"  ERROR: Failed after {max_retries} attempts: {e}")
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 * (attempt + 1)
                logger.info(f"  ERROR downloading data, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"  ERROR: Failed after {max_retries} attempts for {token_id}: {e}")
                return pd.DataFrame(columns=["timestamp", "price", "token_id"])

    return pd.DataFrame(columns=["timestamp", "price", "token_id"])


def download_binance_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    """
    Download Binance futures klines.

    Args:
        symbol: Futures symbol (e.g., BTCUSDT)
        interval: Kline interval (1m, 5m, 1h, etc.)
        start_ts: Start timestamp (Unix milliseconds)
        end_ts: End timestamp (Unix milliseconds)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading Binance klines for {symbol} ({interval})")

    url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"
    all_klines = []

    # Binance API limits to 1000 klines per request
    current_ts = start_ts
    batch_size = 1000

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": batch_size,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            all_klines.extend(klines)

            # Update current_ts to last kline's close time + 1ms
            current_ts = klines[-1][6] + 1

            logger.info(f"  Downloaded {len(all_klines)} klines so far...")
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            logger.info(f"  ERROR downloading klines: {e}")
            break

    # Convert to DataFrame
    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Convert to numeric
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col])

    df["timestamp"] = df["open_time"]
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades_count"]]

    logger.info(f"  Downloaded {len(df)} klines total")
    return df


def download_binance_agg_trades(
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    """
    Download Binance aggregated trades (for CVD calculation).

    Args:
        symbol: Futures symbol
        start_ts: Start timestamp (Unix milliseconds)
        end_ts: End timestamp (Unix milliseconds)

    Returns:
        DataFrame with trade data
    """
    logger.info(f"Downloading Binance aggregated trades for {symbol}")

    url = f"{BINANCE_FUTURES_API}/fapi/v1/aggTrades"
    all_trades = []

    # Note: Binance aggTrades API can be slow for large ranges
    # Consider using historical data downloads for better performance

    logger.info("  WARNING: aggTrades download can be slow. Consider using data snapshots.")
    logger.info("  Skipping for now - implement if needed for CVD calculation")

    return pd.DataFrame()


def download_asset_data(
    asset: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
):
    """
    Download all data for a single asset.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        start_date: Start date
        end_date: End date
        output_dir: Output directory
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading data for {asset}")
    logger.info(f"{'='*60}")

    # # Convert to timestamps
    # start_ts = int(start_date.timestamp())
    # end_ts = int(end_date.timestamp())
    # start_ts_ms = start_ts * 1000
    # end_ts_ms = end_ts * 1000

    # Download Polymarket 15-min market price history FIRST
    logger.info("\n" + "="*60)
    logger.info("Downloading Polymarket 15-min market data")
    logger.info("="*60)

    # Validate market range exists
    market_start_ts, market_end_ts, asset_lower = get_historical_15m_markets(asset, start_date, end_date)

    if market_start_ts == 0:
        logger.info(f"\nNo Polymarket 15-min markets found for {asset} in date range")
    else:
        # Generate all market timestamps
        num_markets = (market_end_ts - market_start_ts) // 900 + 1
        logger.info(f"\nDownloading price history for {num_markets} markets...")

        polymarket_dir = output_dir / "polymarket" / asset
        polymarket_dir.mkdir(parents=True, exist_ok=True)

        # Create output directories
        combined_dir = output_dir / "combined" / asset
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Binance symbol
        binance_symbol = f"{asset}USDT"

        # Process each 15-min market: download both Polymarket and Binance, combine, save
        current_ts = market_start_ts
        idx = 0
        successful_markets = 0

        while current_ts <= market_end_ts:
            slug = f"{asset_lower}-updown-15m-{current_ts}"

            # Get token ID for this specific market
            token_id = get_token_id_for_market(asset_lower, current_ts)

            if not token_id:
                logger.warning(f"  WARNING: Could not find token ID for {slug}, data gap detected, skipping this 15-min window")
                current_ts += 900
                idx += 1
                time.sleep(0.5)  # Rate limit even on failures
                continue

            # Calculate market time range (15 min before end_time)
            market_end_ts_local = current_ts
            market_start_ts_local = current_ts - 900  # 15 minutes

            # Download Polymarket price history
            polymarket_df = download_polymarket_prices(
                token_id=token_id,
                start_ts=market_start_ts_local,
                end_ts=market_end_ts_local,
                fidelity=1,  # 1 minute resolution
            )

            if polymarket_df.empty:
                logger.warning(f"  WARNING: No Polymarket price data for {slug}, skipping this 15-min window")
                current_ts += 900
                idx += 1
                time.sleep(0.5)
                continue

            # Download Binance klines for this 15-min window
            binance_df = download_binance_klines(
                symbol=binance_symbol,
                interval="1m",
                start_ts=market_start_ts_local * 1000,  # Convert to ms
                end_ts=market_end_ts_local * 1000,
            )

            if binance_df.empty:
                logger.warning(f"  WARNING: No Binance klines for {slug}, skipping this 15-min window")
                current_ts += 900
                idx += 1
                continue

            # Combine both datasets
            # Add metadata
            polymarket_df["slug"] = slug
            polymarket_df["market_end_time"] = current_ts
            binance_df["slug"] = slug
            binance_df["market_end_time"] = current_ts

            # # Save combined data for this 15-min market
            # market_file = combined_dir / f"{slug}.parquet"
            # combined_data = {
            #     "polymarket": polymarket_df,
            #     "binance": binance_df,
            # }

            # Save as separate tables in one parquet file
            # For simplicity, we'll save them as separate files
            polymarket_file = combined_dir / f"{slug}_polymarket.parquet"
            binance_file = combined_dir / f"{slug}_binance.parquet"

            polymarket_df.to_parquet(polymarket_file, index=False)
            binance_df.to_parquet(binance_file, index=False)

            successful_markets += 1

            # Progress logging
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{num_markets} markets... ({successful_markets} successful)")

            # Rate limiting - respect Polymarket API limits
            time.sleep(0.5)  # Max 2 requests/second

            current_ts += 900  # Next 15-min market
            idx += 1

        logger.info(f"\n✓ Successfully processed {successful_markets}/{num_markets} markets")
        logger.info(f"  Data saved to: {combined_dir}")



def main():
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH", "SOL", "XRP"],
        help="Assets to download",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical",
        help="Output directory",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir)

    logger.info(f"\n{'='*60}")
    logger.info("HISTORICAL DATA DOWNLOAD")
    logger.info(f"{'='*60}")
    logger.info(f"Assets: {', '.join(args.assets)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}\n")

    # Download data for each asset
    for asset in args.assets:
        download_asset_data(asset, start_date, end_date, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Data saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Run data processing scripts to compute features")
    logger.info("2. Create training episodes from raw data")
    logger.info("3. Start offline training with historical data")


if __name__ == "__main__":
    main()
