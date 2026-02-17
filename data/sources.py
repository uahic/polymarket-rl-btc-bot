"""
Data sources for trading bot.

Provides unified interface for accessing market data from:
- Historical sources (Polymarket + Binance historical APIs, parquet files)
- Live sources (WebSocket streams, REST APIs)

Both sources produce the same RawMarketData format, ensuring
identical preprocessing for training and deployment.
"""

import logging
import sys
import time
import random

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.computer import (
    RawMarketData,
    OrderbookSnapshot,
    FuturesData,
    SpotData,
)
from streams.binance import BinanceStreamer
from streams.binance_futures import FuturesStreamer
from streams.orderbook import OrderbookStreamer

logger = logging.getLogger(__name__)




class DataSource(ABC):
    """
    Abstract data source interface.

    Subclasses must implement methods to:
    - Initialize/reset to start of episode
    - Get current market state
    - Advance to next time step
    """

    @abstractmethod
    def reset(self, **kwargs) -> RawMarketData:
        """
        Initialize new episode.

        Returns:
            Initial market state
        """
        pass

    @abstractmethod
    def get_current(self) -> RawMarketData:
        """
        Get current market observation.

        Returns:
            Current raw market data
        """
        pass

    @abstractmethod
    def advance(self) -> bool:
        """
        Advance to next time step.

        Returns:
            True if more data available, False if episode ended
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Check if episode has ended.

        Returns:
            True if episode is over (market expired, no more data, etc.)
        """
        pass


class HistoricalSource(DataSource):
    """
    Historical data source for offline training.

    Loads pre-downloaded market data from disk and replays it
    tick-by-tick as if it were happening in real-time.

    Data sources:
    - Polymarket price history: /prices-history API
    - Binance futures: klines, aggTrades
    - Cached in parquet files for fast loading

    Directory structure:
        data/historical/
            └── combined/
                └── {asset}/
                    ├── {slug}_polymarket.parquet
                    └── {slug}_binance.parquet

    Where {slug} = "{asset}-updown-15m-{timestamp}"
    Each pair represents one 15-minute market episode.
                └── episodes/
                    └── episode_{id}.parquet  # Pre-computed features
    """

    def __init__(
        self,
        data_dir: str,
        assets: List[str] = ["BTC", "ETH", "SOL", "XRP"],
        episode_length: int = 1800,  # 15 min @ 500ms ticks
        random_start: bool = True,
    ):
        """
        Initialize historical data source.

        Args:
            data_dir: Root directory for historical data
            assets: List of assets to sample episodes from
            episode_length: Number of ticks per episode
            random_start: Whether to randomly sample start time
        """
        self.data_dir = data_dir
        self.assets = assets
        self.episode_length = episode_length
        self.random_start = random_start

        # Current episode state
        self.current_episode: Optional[List[RawMarketData]] = None
        self.current_idx = 0
        self.current_asset = None

    def reset(self, asset: Optional[str] = None, start_time: Optional[datetime] = None) -> RawMarketData:
        """
        Load new episode from historical data.

        Args:
            asset: Specific asset to load (or random if None)
            start_time: Specific start time (or random if None)

        Returns:
            Initial market state
        """
        # Select asset
        if asset is None:
            asset = random.choice(self.assets)
        self.current_asset = asset

        # Load episode from disk
        self.current_episode = self._load_episode(asset, start_time)
        self.current_idx = 0

        return self.current_episode[0]

    def get_current(self) -> RawMarketData:
        """Get current market observation."""
        if self.current_episode is None:
            raise RuntimeError("Must call reset() before get_current()")

        return self.current_episode[self.current_idx]

    def advance(self) -> bool:
        """
        Advance to next tick.

        Returns:
            True if more data available
        """
        self.current_idx += 1
        return self.current_idx < len(self.current_episode)

    def is_done(self) -> bool:
        """Check if episode ended."""
        if self.current_episode is None:
            return True
        return self.current_idx >= len(self.current_episode)

    def _load_episode(self, asset: str, start_time: Optional[datetime] = None) -> List[RawMarketData]:
        """
        Load historical episode from combined parquet files.

        New structure: Each 15-min market has separate files:
          - {slug}_polymarket.parquet
          - {slug}_binance.parquet

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            start_time: Episode start time (or random if None)

        Returns:
            List of RawMarketData observations
        """
        from pathlib import Path
        import pandas as pd

        # Load data files from combined directory
        data_path = Path(self.data_dir)
        combined_dir = data_path / "combined" / asset

        if not combined_dir.exists():
            logger.warning(f"[HistoricalSource] No combined data found for {asset} at {combined_dir}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        # Find all available markets
        polymarket_files = list(combined_dir.glob("*_polymarket.parquet"))

        if not polymarket_files:
            logger.warning(f"[HistoricalSource] No market data found in {combined_dir}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        # Randomly select a market for this episode
        selected_file = random.choice(polymarket_files)
        slug = selected_file.stem.replace("_polymarket", "")

        binance_file = combined_dir / f"{slug}_binance.parquet"

        if not binance_file.exists():
            logger.warning(f"[HistoricalSource] Missing Binance data for {slug}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        logger.info(f"[HistoricalSource] Loading episode: {slug}")

        try:
            df_polymarket = pd.read_parquet(selected_file)
            df_klines = pd.read_parquet(binance_file)
        except Exception as e:
            logger.error(f"[HistoricalSource] ERROR loading data: {e}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        # Validate dataframes
        if df_polymarket.empty or df_klines.empty:
            logger.warning(f"[HistoricalSource] Empty dataframes for {slug}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        # Validate required columns
        required_klines_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df_klines.columns for col in required_klines_cols):
            logger.error("[HistoricalSource] Missing columns in Binance data")
            logger.error(f"[HistoricalSource] Found: {df_klines.columns.tolist()}")
            logger.warning("[HistoricalSource] Falling back to dummy data")
            return self._generate_dummy_episode(asset)

        # Use klines as base
        df = df_klines
        episode_df = df
        polymarket_episode = None

        if df_polymarket is not None and start_time is None:
            # Try to sample from a complete Polymarket market
            available_markets = df_polymarket.groupby('slug')
            market_slugs = list(available_markets.groups.keys())

            # Shuffle and try to find a good market
            np.random.shuffle(market_slugs)

            for slug in market_slugs[:10]:  # Try up to 10 markets
                market_data = available_markets.get_group(slug)

                # Check if this market has enough data points (at least 10 minutes)
                if len(market_data) < 10:
                    continue

                # Get the time range for this market
                market_start_ts = market_data['timestamp'].min()
                market_end_ts = market_data['timestamp'].max()

                # Find matching Binance klines (1-min resolution, so ~15 points for 15-min market)
                # We need 1800 ticks at 500ms = 900 seconds = 15 minutes
                # But Binance has 1-min bars, so we need 15 bars
                binance_mask = (df['timestamp'] >= market_start_ts) & (df['timestamp'] <= market_end_ts)
                binance_for_market = df[binance_mask]

                if len(binance_for_market) >= 10:  # Need at least 10 minutes
                    episode_df = binance_for_market.iloc[:15].reset_index(drop=True)  # Take 15 bars (15 min)
                    polymarket_episode = market_data.reset_index(drop=True)
                    logger.info(f"[HistoricalSource] Selected Polymarket market: {slug}")
                    logger.info(f"[HistoricalSource] Binance bars: {len(episode_df)}, Polymarket points: {len(polymarket_episode)}")
                    break

        # Fallback to random Binance sampling
        if episode_df is None:
            logger.info("[HistoricalSource] Using random Binance klines (no Polymarket alignment)")
            # Sample 15 consecutive 1-min bars for a 15-min episode
            required_bars = 15
            max_start_idx = len(df) - required_bars
            if max_start_idx <= 0:
                logger.warning("[HistoricalSource] Not enough data for episode")
                return self._generate_dummy_episode(asset)

            start_idx = np.random.randint(0, max_start_idx)
            episode_df = df.iloc[start_idx:start_idx + required_bars].reset_index(drop=True)
            polymarket_episode = None

        if len(episode_df) < 10:
            logger.warning(f"[HistoricalSource] Episode too short ({len(episode_df)} bars)")
            return self._generate_dummy_episode(asset)

        # Build RawMarketData for each tick at 500ms resolution
        # We have 1-min Binance bars, need to interpolate to 500ms ticks
        # Each 1-min bar = 120 ticks at 500ms
        # For 15 bars = 1800 ticks total
        episode = []
        prob_history = []

        # Pre-index Polymarket data by timestamp for faster lookup
        polymarket_lookup = {}
        if polymarket_episode is not None:
            for _, pm_row in polymarket_episode.iterrows():
                ts_key = int(pm_row['timestamp'] // 1000)  # Round to seconds
                polymarket_lookup[ts_key] = float(pm_row['price'])

        for bar_idx, row in episode_df.iterrows():
            # Each bar represents 60 seconds, create 120 ticks (500ms each)
            bar_start_ts = int(row.timestamp)

            # For simplicity, repeat the same bar data for all 120 sub-ticks
            # A more sophisticated approach would interpolate OHLC
            for sub_tick in range(120):
                tick_ts_ms = bar_start_ts + (sub_tick * 500)  # Add 500ms per tick
                tick_idx = bar_idx * 120 + sub_tick
                # Compute returns (only from bar data, not sub-ticks)
                returns_1m = self._compute_returns(episode_df, bar_idx, window=1)
                returns_5m = self._compute_returns(episode_df, bar_idx, window=5)
                returns_10m = self._compute_returns(episode_df, bar_idx, window=10)

                # Compute volatility
                realized_vol_5m = self._compute_volatility(episode_df, bar_idx, window=5)
                avg_vol = self._compute_volatility(episode_df, bar_idx, window=30)

                # Compute CVD
                cvd = float(row.volume) * np.sign(row.close - row.open)
                cvd_history = [cvd * 0.9, cvd] if bar_idx > 0 else [0.0, cvd]

                # Trade intensity
                trade_intensity = float(row.get("trades_count", 100)) / 60.0

                # Get Polymarket price for this tick
                tick_ts_sec = tick_ts_ms // 1000
                prob_up = polymarket_lookup.get(tick_ts_sec, 0.5)  # Fast lookup

                # Construct orderbook
                spread = 0.01
                best_bid = max(0.01, prob_up - spread / 2)
                best_ask = min(0.99, prob_up + spread / 2)

                # Create RawMarketData for this 500ms tick
                raw_data = RawMarketData(
                    timestamp=float(tick_ts_ms) / 1000.0,  # Convert to seconds
                    asset=asset,
                    orderbook=OrderbookSnapshot(
                        timestamp=float(tick_ts_ms) / 1000.0,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        spread=spread,
                        bids_l5=[(best_bid, 100.0)],
                        asks_l5=[(best_ask, 100.0)],
                    ),
                    futures=FuturesData(
                        timestamp=float(tick_ts_ms) / 1000.0,
                        price=float(row.close),
                        returns_1m=returns_1m,
                        returns_5m=returns_5m,
                        returns_10m=returns_10m,
                        cvd=cvd,
                        cvd_history=cvd_history,
                        trade_flow_imbalance=0.0,
                        trade_intensity=trade_intensity,
                        large_trade_flag=1.0 if row.volume > episode_df.volume.quantile(0.95) else 0.0,
                        realized_vol_5m=realized_vol_5m,
                        avg_vol=avg_vol if avg_vol > 0 else realized_vol_5m,
                    ),
                    spot=SpotData(
                        timestamp=float(tick_ts_ms) / 1000.0,
                        price=float(row.close),
                        change_pct=returns_1m,
                    ),
                    prob_up=prob_up,
                    time_remaining=1.0 - (tick_idx / self.episode_length),
                    prob_history=list(prob_history),
                    vol_regime=1.0 if realized_vol_5m > avg_vol else 0.0,
                    trend_regime=1.0 if returns_10m > 0.001 else 0.0,
                )

                episode.append(raw_data)
                prob_history.append(prob_up)

                # Keep history bounded
                if len(prob_history) > 50:
                    prob_history = prob_history[-50:]

                # Stop if we've generated enough ticks
                if len(episode) >= self.episode_length:
                    break

            if len(episode) >= self.episode_length:
                break

        logger.info(f"[HistoricalSource] Loaded {len(episode)} ticks for {asset}")
        return episode

    def _compute_returns(self, df: 'pd.DataFrame', idx: int, window: int) -> float:
        """Compute returns over window."""
        if idx < window:
            return 0.0

        current_price = df.iloc[idx].close
        past_price = df.iloc[idx - window].close

        if past_price == 0:
            return 0.0

        return float((current_price - past_price) / past_price)

    def _compute_volatility(self, df: 'pd.DataFrame', idx: int, window: int) -> float:
        """Compute realized volatility."""
        if idx < window:
            return 0.0

        prices = df.iloc[max(0, idx - window):idx + 1].close.values
        if len(prices) < 2:
            return 0.0

        log_returns = np.diff(np.log(prices + 1e-8))  # Add small epsilon to avoid log(0)
        return float(np.std(log_returns) * np.sqrt(252 * 24 * 60))  # Annualized

    def _generate_dummy_episode(self, asset: str) -> List[RawMarketData]:
        """Generate dummy episode for testing when no data available."""
        logger.info(f"[HistoricalSource] Generating dummy episode for {asset}")

        episode = []
        prob_history = []

        for tick in range(self.episode_length):
            # Simulate some market dynamics
            t = tick * 0.5  # 500ms per tick
            prob = 0.5 + 0.1 * np.sin(t / 100)  # Oscillating probability
            prob_history.append(prob)

            # Keep only last 50 ticks
            if len(prob_history) > 50:
                prob_history = prob_history[-50:]

            raw_data = RawMarketData(
                timestamp=time.time() + tick * 0.5,
                asset=asset,
                orderbook=OrderbookSnapshot(
                    timestamp=time.time() + tick * 0.5,
                    best_bid=prob - 0.005,
                    best_ask=prob + 0.005,
                    spread=0.01,
                    bids_l5=[(prob - 0.005, 100.0)],
                    asks_l5=[(prob + 0.005, 100.0)],
                ),
                futures=FuturesData(
                    timestamp=time.time() + tick * 0.5,
                    price=50000.0,
                    returns_1m=0.001 * np.random.randn(),
                    returns_5m=0.002 * np.random.randn(),
                    returns_10m=0.003 * np.random.randn(),
                    cvd=tick * 10.0,
                    cvd_history=[max(0, tick - 1) * 10.0, tick * 10.0],
                    trade_flow_imbalance=0.1 * np.random.randn(),
                    trade_intensity=5.0,
                    large_trade_flag=0.0,
                    realized_vol_5m=0.02,
                    avg_vol=0.02,
                ),
                spot=SpotData(
                    timestamp=time.time() + tick * 0.5,
                    price=50000.0,
                    change_pct=0.0,
                ),
                prob_up=prob,
                time_remaining=1.0 - (tick / self.episode_length),
                prob_history=list(prob_history),
                vol_regime=0.0,
                trend_regime=0.0,
            )
            episode.append(raw_data)

        return episode


class LiveSource(DataSource):
    """
    Live data source for real-time trading.

    Wraps existing stream infrastructure:
    - OrderbookStreamer (Polymarket WebSocket)
    - BinanceStreamer (spot prices)
    - FuturesStreamer (Binance futures data)

    Aggregates current state from all streams and packages
    it into RawMarketData format.
    """

    def __init__(
        self,
        orderbook_streamer: OrderbookStreamer,
        binance_streamer: BinanceStreamer,
        futures_streamer: FuturesStreamer,
        tick_interval: float = 0.5,  # 500ms
    ):
        """
        Initialize live data source.

        Args:
            orderbook_streamer: Polymarket orderbook WebSocket stream
            binance_streamer: Binance spot price stream
            futures_streamer: Binance futures data stream
            tick_interval: Time between ticks in seconds
        """
        self.orderbook_streamer = orderbook_streamer
        self.binance_streamer = binance_streamer
        self.futures_streamer = futures_streamer
        self.tick_interval = tick_interval

        # Current market state
        self.current_asset = None
        self.current_market = None
        self.episode_start_time = None
        self.episode_duration = 900.0  # 15 minutes

        # History tracking
        self.prob_history: List[float] = []

    def reset(self, asset: str, market_id: str) -> RawMarketData:
        """
        Initialize new live trading episode.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            market_id: Polymarket market/token ID

        Returns:
            Initial market state
        """
        self.current_asset = asset
        self.current_market = market_id
        self.prob_history = []

        # Wait for streams to have data
        logger.info(f"[LiveSource] Waiting for stream data for {asset}...")
        time.sleep(1.0)

        # Start episode timer after waiting for stream data
        self.episode_start_time = time.time()

        return self.get_current()

    def get_current(self) -> RawMarketData:
        """
        Aggregate current state from all streams.

        Pulls latest data from:
        - Orderbook streamer
        - Binance spot streamer
        - Futures streamer

        And packages into RawMarketData.
        """
        timestamp = time.time()

        # Get orderbook data
        ob_data = self.orderbook_streamer.get_latest(self.current_market)
        orderbook = OrderbookSnapshot(
            timestamp=timestamp,
            best_bid=ob_data.get("best_bid", 0.0),
            best_ask=ob_data.get("best_ask", 0.0),
            spread=ob_data.get("spread", 0.0),
            bids_l5=ob_data.get("bids_l5", []),
            asks_l5=ob_data.get("asks_l5", []),
        )

        # Get futures data
        futures_data = self.futures_streamer.get_latest(self.current_asset)
        futures = FuturesData(
            timestamp=timestamp,
            price=futures_data.get("price", 0.0),
            returns_1m=futures_data.get("returns_1m", 0.0),
            returns_5m=futures_data.get("returns_5m", 0.0),
            returns_10m=futures_data.get("returns_10m", 0.0),
            cvd=futures_data.get("cvd", 0.0),
            cvd_history=futures_data.get("cvd_history", []),
            trade_flow_imbalance=futures_data.get("trade_flow_imbalance", 0.0),
            trade_intensity=futures_data.get("trade_intensity", 0.0),
            large_trade_flag=futures_data.get("large_trade_flag", 0.0),
            realized_vol_5m=futures_data.get("realized_vol_5m", 0.0),
            avg_vol=futures_data.get("avg_vol", 0.0),
        )

        # Get spot data
        spot_data = self.binance_streamer.get_latest(self.current_asset)
        spot = SpotData(
            timestamp=timestamp,
            price=spot_data.get("price", 0.0),
            change_pct=0.0,  # Not used in features
        )

        # Calculate time remaining
        elapsed = timestamp - self.episode_start_time
        time_remaining = max(0.0, 1.0 - (elapsed / self.episode_duration))

        # Current probability (from orderbook mid price), clamped to valid range
        prob_up = max(0.0, min(1.0, orderbook.mid_price))

        # Update history
        self.prob_history.append(prob_up)
        if len(self.prob_history) > 50:
            self.prob_history = self.prob_history[-50:]

        return RawMarketData(
            timestamp=timestamp,
            asset=self.current_asset,
            orderbook=orderbook,
            futures=futures,
            spot=spot,
            prob_up=prob_up,
            time_remaining=time_remaining,
            prob_history=list(self.prob_history),
            vol_regime=futures_data.get("vol_regime", 0.0),
            trend_regime=futures_data.get("trend_regime", 0.0),
        )

    def advance(self) -> bool:
        """
        Wait for next tick (500ms).

        In live trading, this just waits for the next time interval.

        Returns:
            True (always has more data unless market expires)
        """
        time.sleep(self.tick_interval)
        return not self.is_done()

    def is_done(self) -> bool:
        """
        Check if market has expired.

        Returns:
            True if 15 minutes elapsed
        """
        if self.episode_start_time is None:
            return True

        elapsed = time.time() - self.episode_start_time
        return elapsed >= self.episode_duration
