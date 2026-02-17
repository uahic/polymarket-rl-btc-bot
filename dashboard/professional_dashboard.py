#!/usr/bin/env python3
"""
Professional Hedge Fund Style Dashboard for RL Trading Bot

Features:
- Real-time PnL tracking and equity curve
- Training metrics (policy loss, value loss, entropy, KL divergence)
- Expected PnL predictions over time
- Detailed trade history with analytics
- Win rate, Sharpe ratio, max drawdown
- Market positions and exposure
- Performance attribution by asset
"""

import logging
import threading
import time
import json
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Deque
from collections import deque
from pathlib import Path
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
import numpy as np

logger = logging.getLogger(__name__)


class DashboardState:
    """Central state management for dashboard."""

    def __init__(self):
        # Trading metrics
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_volume = 0.0

        # Position tracking
        self.positions: Dict[str, dict] = {}
        self.markets: Dict[str, dict] = {}

        # Historical data (last 1000 points)
        self.pnl_history: Deque = deque(maxlen=1000)
        self.expected_pnl_history: Deque = deque(maxlen=1000)
        self.timestamp_history: Deque = deque(maxlen=1000)

        # Training metrics history
        self.policy_loss_history: Deque = deque(maxlen=500)
        self.value_loss_history: Deque = deque(maxlen=500)
        self.entropy_history: Deque = deque(maxlen=500)
        self.kl_divergence_history: Deque = deque(maxlen=500)
        self.clip_fraction_history: Deque = deque(maxlen=500)
        self.explained_variance_history: Deque = deque(maxlen=500)

        # Trade history
        self.recent_trades: Deque = deque(maxlen=100)

        # Performance metrics
        self.returns_series = []
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0

        # RL training state
        self.buffer_size = 0
        self.max_buffer_size = 2048
        self.update_count = 0
        self.episode_count = 0
        self.avg_episode_reward = 0.0
        self.avg_episode_length = 0.0

        # Expected value tracking
        self.expected_value = 0.0
        self.value_error = 0.0

        # Performance by asset
        self.asset_performance: Dict[str, dict] = {}


dashboard_state = DashboardState()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hedge-fund-dashboard-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RL Trading Dashboard - Professional</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #0a0b0d;
            --bg-secondary: #13151a;
            --bg-tertiary: #1a1d24;
            --border: #252932;
            --text-primary: #e8eaed;
            --text-secondary: #9aa0a6;
            --text-dim: #5f6368;
            --green: #34a853;
            --green-bg: rgba(52, 168, 83, 0.1);
            --red: #ea4335;
            --red-bg: rgba(234, 67, 53, 0.1);
            --blue: #4285f4;
            --blue-bg: rgba(66, 133, 244, 0.1);
            --amber: #fbbc04;
            --amber-bg: rgba(251, 188, 4, 0.1);
            --purple: #9334e6;
            --shadow: rgba(0, 0, 0, 0.3);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            overflow-x: hidden;
        }

        .container {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            grid-template-rows: 80px 1fr;
            height: 100vh;
            gap: 0;
        }

        /* Header */
        .header {
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .brand h1 {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }

        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: var(--green-bg);
            border: 1px solid var(--green);
            border-radius: 6px;
        }

        .live-dot {
            width: 6px;
            height: 6px;
            background: var(--green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.9); }
        }

        .live-text {
            font-size: 11px;
            font-weight: 500;
            color: var(--green);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .header-stats {
            display: flex;
            gap: 40px;
        }

        .header-stat {
            text-align: right;
        }

        .header-stat-value {
            font-size: 28px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.5px;
        }

        .header-stat-value.positive { color: var(--green); }
        .header-stat-value.negative { color: var(--red); }
        .header-stat-value.neutral { color: var(--text-primary); }

        .header-stat-label {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }

        /* Sidebar - Metrics */
        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            padding: 24px 20px;
        }

        .metric-section {
            margin-bottom: 32px;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
        }

        .metric-label {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .metric-value {
            font-size: 14px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            color: var(--text-primary);
        }

        .metric-value.small {
            font-size: 13px;
        }

        .metric-value.green { color: var(--green); }
        .metric-value.red { color: var(--red); }
        .metric-value.amber { color: var(--amber); }

        /* Main area */
        .main {
            background: var(--bg-primary);
            overflow-y: auto;
            padding: 24px;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .chart-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px var(--shadow);
        }

        .chart-card.full-width {
            grid-column: 1 / -1;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .chart-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-subtitle {
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 2px;
        }

        .chart-legend {
            display: flex;
            gap: 16px;
            font-size: 11px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-secondary);
        }

        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }

        .chart-container {
            position: relative;
            height: 280px;
        }

        .chart-container.tall {
            height: 400px;
        }

        /* Right panel - Trades */
        .right-panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .panel-header {
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }

        .panel-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .panel-subtitle {
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 4px;
        }

        .trades-container {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }

        .trade-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            transition: all 0.2s;
        }

        .trade-card:hover {
            border-color: var(--text-dim);
        }

        .trade-card.win {
            border-left: 3px solid var(--green);
        }

        .trade-card.loss {
            border-left: 3px solid var(--red);
        }

        .trade-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .trade-asset {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .trade-time {
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-dim);
        }

        .trade-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 11px;
        }

        .trade-detail-label {
            color: var(--text-dim);
        }

        .trade-detail-value {
            font-weight: 500;
            color: var(--text-secondary);
            font-variant-numeric: tabular-nums;
        }

        .trade-pnl {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .trade-pnl-label {
            font-size: 11px;
            color: var(--text-dim);
        }

        .trade-pnl-value {
            font-size: 15px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        .trade-pnl-value.positive { color: var(--green); }
        .trade-pnl-value.negative { color: var(--red); }

        .trade-side {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .trade-side.long {
            background: var(--green-bg);
            color: var(--green);
        }

        .trade-side.short {
            background: var(--red-bg);
            color: var(--red);
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-dim);
        }

        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
            opacity: 0.3;
        }

        .empty-state-text {
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="brand">
                <h1>RL Trading Dashboard</h1>
                <div class="live-badge">
                    <div class="live-dot"></div>
                    <span class="live-text">Live</span>
                </div>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-value neutral" id="total-trades">0</div>
                    <div class="header-stat-label">Total Trades</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value neutral" id="win-rate">0%</div>
                    <div class="header-stat-label">Win Rate</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value positive" id="total-pnl">$0.00</div>
                    <div class="header-stat-label">Total PnL</div>
                </div>
            </div>
        </header>

        <!-- Left Sidebar -->
        <aside class="sidebar">
            <div class="metric-section">
                <div class="section-title">Performance</div>
                <div class="metric-row">
                    <span class="metric-label">Realized PnL</span>
                    <span class="metric-value" id="realized-pnl">$0.00</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Unrealized PnL</span>
                    <span class="metric-value" id="unrealized-pnl">$0.00</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Sharpe Ratio</span>
                    <span class="metric-value" id="sharpe-ratio">0.00</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Max Drawdown</span>
                    <span class="metric-value red" id="max-drawdown">0.00%</span>
                </div>
            </div>

            <div class="metric-section">
                <div class="section-title">Training State</div>
                <div class="metric-row">
                    <span class="metric-label">Episodes</span>
                    <span class="metric-value small" id="episodes">0</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Updates</span>
                    <span class="metric-value small" id="updates">0</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Buffer Size</span>
                    <span class="metric-value small" id="buffer-size">0/0</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Episode Reward</span>
                    <span class="metric-value small" id="avg-episode-reward">0.00</span>
                </div>
            </div>

            <div class="metric-section">
                <div class="section-title">Latest Metrics</div>
                <div class="metric-row">
                    <span class="metric-label">Policy Loss</span>
                    <span class="metric-value small amber" id="policy-loss">—</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Value Loss</span>
                    <span class="metric-value small amber" id="value-loss">—</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Entropy</span>
                    <span class="metric-value small" id="entropy">—</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">KL Divergence</span>
                    <span class="metric-value small" id="kl-divergence">—</span>
                </div>
            </div>

            <div class="metric-section">
                <div class="section-title">Value Estimation</div>
                <div class="metric-row">
                    <span class="metric-label">Expected Value</span>
                    <span class="metric-value small" id="expected-value">$0.00</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Value Error</span>
                    <span class="metric-value small" id="value-error">0.00</span>
                </div>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="main">
            <div class="chart-grid">
                <!-- Equity Curve -->
                <div class="chart-card full-width">
                    <div class="chart-header">
                        <div>
                            <div class="chart-title">Equity Curve & Expected PnL</div>
                            <div class="chart-subtitle">Real-time vs. predicted performance</div>
                        </div>
                        <div class="chart-legend">
                            <div class="legend-item">
                                <div class="legend-dot" style="background: var(--green);"></div>
                                <span>Actual PnL</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-dot" style="background: var(--blue); opacity: 0.6;"></div>
                                <span>Expected PnL</span>
                            </div>
                        </div>
                    </div>
                    <div class="chart-container tall">
                        <canvas id="equity-chart"></canvas>
                    </div>
                </div>

                <!-- Policy & Value Loss -->
                <div class="chart-card">
                    <div class="chart-header">
                        <div>
                            <div class="chart-title">Training Losses</div>
                            <div class="chart-subtitle">Policy and value function</div>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="loss-chart"></canvas>
                    </div>
                </div>

                <!-- Entropy & KL Divergence -->
                <div class="chart-card">
                    <div class="chart-header">
                        <div>
                            <div class="chart-title">Exploration Metrics</div>
                            <div class="chart-subtitle">Entropy and KL divergence</div>
                        </div>
                    </div>
                    <div class="chart-container">
                        <canvas id="exploration-chart"></canvas>
                    </div>
                </div>
            </div>
        </main>

        <!-- Right Panel - Trades -->
        <aside class="right-panel">
            <div class="panel-header">
                <div class="panel-title">Recent Trades</div>
                <div class="panel-subtitle">Last 100 executions</div>
            </div>
            <div class="trades-container" id="trades-container">
                <div class="empty-state">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-text">Waiting for trades...</div>
                </div>
            </div>
        </aside>
    </div>

    <script>
        const socket = io();

        // Chart configurations
        const chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(26, 29, 36, 0.95)',
                    titleColor: '#e8eaed',
                    bodyColor: '#9aa0a6',
                    borderColor: '#252932',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    titleFont: { size: 12, weight: '600' },
                    bodyFont: { size: 11 }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(37, 41, 50, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#5f6368',
                        font: { size: 10 }
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    grid: {
                        color: 'rgba(37, 41, 50, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#5f6368',
                        font: { size: 10 }
                    }
                }
            }
        };

        // Initialize charts
        let equityChart, lossChart, explorationChart;

        function initCharts() {
            // Equity curve
            const equityCtx = document.getElementById('equity-chart').getContext('2d');
            equityChart = new Chart(equityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Actual PnL',
                            data: [],
                            borderColor: '#34a853',
                            backgroundColor: 'rgba(52, 168, 83, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0
                        },
                        {
                            label: 'Expected PnL',
                            data: [],
                            borderColor: 'rgba(66, 133, 244, 0.6)',
                            backgroundColor: 'rgba(66, 133, 244, 0.05)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    ...chartDefaults,
                    scales: {
                        ...chartDefaults.scales,
                        y: {
                            ...chartDefaults.scales.y,
                            ticks: {
                                ...chartDefaults.scales.y.ticks,
                                callback: (v) => '$' + v.toFixed(2)
                            }
                        }
                    }
                }
            });

            // Training losses
            const lossCtx = document.getElementById('loss-chart').getContext('2d');
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Policy Loss',
                            data: [],
                            borderColor: '#fbbc04',
                            borderWidth: 2,
                            tension: 0.3,
                            pointRadius: 0
                        },
                        {
                            label: 'Value Loss',
                            data: [],
                            borderColor: '#ea4335',
                            borderWidth: 2,
                            tension: 0.3,
                            pointRadius: 0
                        }
                    ]
                },
                options: chartDefaults
            });

            // Exploration metrics
            const explorationCtx = document.getElementById('exploration-chart').getContext('2d');
            explorationChart = new Chart(explorationCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Entropy',
                            data: [],
                            borderColor: '#9334e6',
                            borderWidth: 2,
                            tension: 0.3,
                            pointRadius: 0,
                            yAxisID: 'y'
                        },
                        {
                            label: 'KL Divergence',
                            data: [],
                            borderColor: '#4285f4',
                            borderWidth: 2,
                            tension: 0.3,
                            pointRadius: 0,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    ...chartDefaults,
                    scales: {
                        x: chartDefaults.scales.x,
                        y: {
                            ...chartDefaults.scales.y,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Entropy',
                                color: '#9334e6',
                                font: { size: 10 }
                            }
                        },
                        y1: {
                            ...chartDefaults.scales.y,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'KL Divergence',
                                color: '#4285f4',
                                font: { size: 10 }
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            });
        }

        function updateElement(id, value, colorClass = null) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
                if (colorClass) {
                    el.className = el.className.split(' ')[0] + ' ' + colorClass;
                }
            }
        }

        function formatPnL(value) {
            const sign = value >= 0 ? '+' : '';
            return sign + '$' + value.toFixed(2);
        }

        function formatPercent(value) {
            return value.toFixed(2) + '%';
        }

        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to dashboard');
            initCharts();
        });

        socket.on('metrics_update', (data) => {
            // Header stats
            updateElement('total-trades', data.total_trades);
            updateElement('win-rate', formatPercent(data.win_rate),
                data.win_rate >= 50 ? 'positive' : data.win_rate > 0 ? 'negative' : 'neutral');
            updateElement('total-pnl', formatPnL(data.total_pnl),
                data.total_pnl >= 0 ? 'positive' : 'negative');

            // Sidebar metrics
            updateElement('realized-pnl', formatPnL(data.realized_pnl));
            updateElement('unrealized-pnl', formatPnL(data.unrealized_pnl));
            updateElement('sharpe-ratio', data.sharpe_ratio.toFixed(2));
            updateElement('max-drawdown', formatPercent(data.max_drawdown));

            updateElement('episodes', data.episode_count);
            updateElement('updates', data.update_count);
            updateElement('buffer-size', data.buffer_size + '/' + data.max_buffer_size);
            updateElement('avg-episode-reward', data.avg_episode_reward.toFixed(2));

            updateElement('expected-value', formatPnL(data.expected_value));
            updateElement('value-error', data.value_error.toFixed(4));

            // Update equity curve
            if (data.pnl_history && data.pnl_history.length > 0) {
                equityChart.data.labels = data.pnl_history.map((_, i) => i);
                equityChart.data.datasets[0].data = data.pnl_history;
                equityChart.data.datasets[1].data = data.expected_pnl_history || [];
                equityChart.update('none');
            }
        });

        socket.on('training_update', (data) => {
            // Update latest metrics
            if (data.policy_loss !== undefined) {
                updateElement('policy-loss', data.policy_loss.toFixed(4));
            }
            if (data.value_loss !== undefined) {
                updateElement('value-loss', data.value_loss.toFixed(4));
            }
            if (data.entropy !== undefined) {
                updateElement('entropy', data.entropy.toFixed(4));
            }
            if (data.kl_divergence !== undefined) {
                updateElement('kl-divergence', data.kl_divergence.toFixed(6));
            }

            // Update loss chart
            if (data.policy_loss_history && data.policy_loss_history.length > 0) {
                lossChart.data.labels = data.policy_loss_history.map((_, i) => i);
                lossChart.data.datasets[0].data = data.policy_loss_history;
                lossChart.data.datasets[1].data = data.value_loss_history || [];
                lossChart.update('none');
            }

            // Update exploration chart
            if (data.entropy_history && data.entropy_history.length > 0) {
                explorationChart.data.labels = data.entropy_history.map((_, i) => i);
                explorationChart.data.datasets[0].data = data.entropy_history;
                explorationChart.data.datasets[1].data = data.kl_divergence_history || [];
                explorationChart.update('none');
            }
        });

        socket.on('trade', (trade) => {
            const container = document.getElementById('trades-container');

            // Remove empty state if present
            const emptyState = container.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }

            // Create trade card
            const card = document.createElement('div');
            const winClass = trade.pnl >= 0 ? 'win' : 'loss';
            const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
            const sideClass = trade.side.toLowerCase();

            card.className = 'trade-card ' + winClass;
            card.innerHTML = `
                <div class="trade-header">
                    <div>
                        <span class="trade-asset">${trade.asset}</span>
                        <span class="trade-side ${sideClass}">${trade.side}</span>
                    </div>
                    <span class="trade-time">${trade.timestamp}</span>
                </div>
                <div class="trade-details">
                    <div>
                        <div class="trade-detail-label">Entry</div>
                        <div class="trade-detail-value">$${trade.entry_price.toFixed(4)}</div>
                    </div>
                    <div>
                        <div class="trade-detail-label">Exit</div>
                        <div class="trade-detail-value">$${trade.exit_price.toFixed(4)}</div>
                    </div>
                    <div>
                        <div class="trade-detail-label">Size</div>
                        <div class="trade-detail-value">$${trade.size.toFixed(2)}</div>
                    </div>
                    <div>
                        <div class="trade-detail-label">Duration</div>
                        <div class="trade-detail-value">${Math.floor(trade.duration_sec)}s</div>
                    </div>
                </div>
                <div class="trade-pnl">
                    <span class="trade-pnl-label">Profit/Loss</span>
                    <span class="trade-pnl-value ${pnlClass}">${formatPnL(trade.pnl)}</span>
                </div>
            `;

            // Insert at top
            container.insertBefore(card, container.firstChild);

            // Keep only last 100
            while (container.children.length > 100) {
                container.removeChild(container.lastChild);
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve dashboard HTML."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    """Get current dashboard state as JSON."""
    return jsonify({
        'total_pnl': dashboard_state.total_pnl,
        'realized_pnl': dashboard_state.realized_pnl,
        'unrealized_pnl': dashboard_state.unrealized_pnl,
        'total_trades': dashboard_state.total_trades,
        'win_rate': dashboard_state.win_rate,
        'sharpe_ratio': dashboard_state.sharpe_ratio,
        'max_drawdown': dashboard_state.max_drawdown,
    })


def emit_metrics():
    """Emit current metrics to all connected clients."""
    socketio.emit('metrics_update', {
        'total_pnl': dashboard_state.total_pnl,
        'realized_pnl': dashboard_state.realized_pnl,
        'unrealized_pnl': dashboard_state.unrealized_pnl,
        'total_trades': dashboard_state.total_trades,
        'winning_trades': dashboard_state.winning_trades,
        'losing_trades': dashboard_state.losing_trades,
        'win_rate': dashboard_state.win_rate,
        'sharpe_ratio': dashboard_state.sharpe_ratio,
        'max_drawdown': dashboard_state.max_drawdown,
        'episode_count': dashboard_state.episode_count,
        'update_count': dashboard_state.update_count,
        'buffer_size': dashboard_state.buffer_size,
        'max_buffer_size': dashboard_state.max_buffer_size,
        'avg_episode_reward': dashboard_state.avg_episode_reward,
        'expected_value': dashboard_state.expected_value,
        'value_error': dashboard_state.value_error,
        'pnl_history': list(dashboard_state.pnl_history),
        'expected_pnl_history': list(dashboard_state.expected_pnl_history),
    })


def emit_training_metrics():
    """Emit training metrics to all connected clients."""
    socketio.emit('training_update', {
        'policy_loss': list(dashboard_state.policy_loss_history)[-1] if dashboard_state.policy_loss_history else 0,
        'value_loss': list(dashboard_state.value_loss_history)[-1] if dashboard_state.value_loss_history else 0,
        'entropy': list(dashboard_state.entropy_history)[-1] if dashboard_state.entropy_history else 0,
        'kl_divergence': list(dashboard_state.kl_divergence_history)[-1] if dashboard_state.kl_divergence_history else 0,
        'policy_loss_history': list(dashboard_state.policy_loss_history),
        'value_loss_history': list(dashboard_state.value_loss_history),
        'entropy_history': list(dashboard_state.entropy_history),
        'kl_divergence_history': list(dashboard_state.kl_divergence_history),
        'clip_fraction_history': list(dashboard_state.clip_fraction_history),
        'explained_variance_history': list(dashboard_state.explained_variance_history),
    })


def metrics_emitter():
    """Background thread to emit metrics periodically."""
    while True:
        time.sleep(0.5)
        emit_metrics()
        if dashboard_state.update_count > 0:
            emit_training_metrics()


# ============================================================================
# Public API for updating dashboard from training loop
# ============================================================================

def update_pnl(total_pnl: float, realized_pnl: float = None, unrealized_pnl: float = None):
    """Update PnL values."""
    dashboard_state.total_pnl = total_pnl
    if realized_pnl is not None:
        dashboard_state.realized_pnl = realized_pnl
    if unrealized_pnl is not None:
        dashboard_state.unrealized_pnl = unrealized_pnl

    dashboard_state.pnl_history.append(total_pnl)
    dashboard_state.timestamp_history.append(datetime.now().isoformat())

    # Update returns series for Sharpe ratio
    if len(dashboard_state.pnl_history) > 1:
        returns = total_pnl - dashboard_state.pnl_history[-2]
        dashboard_state.returns_series.append(returns)

        # Calculate Sharpe ratio
        if len(dashboard_state.returns_series) > 10:
            returns_array = np.array(dashboard_state.returns_series[-100:])
            if returns_array.std() > 0:
                dashboard_state.sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252)

        # Calculate max drawdown
        peak = max(dashboard_state.pnl_history)
        if peak > 0:
            drawdown = (total_pnl - peak) / peak * 100
            dashboard_state.max_drawdown = min(dashboard_state.max_drawdown, drawdown)


def update_expected_pnl(expected_pnl: float):
    """Update expected PnL prediction."""
    dashboard_state.expected_pnl_history.append(expected_pnl)


def log_trade(
    asset: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size: float,
    pnl: float,
    duration_sec: float,
    timestamp: str = None
):
    """Log a completed trade."""
    dashboard_state.total_trades += 1

    if pnl > 0:
        dashboard_state.winning_trades += 1
    elif pnl < 0:
        dashboard_state.losing_trades += 1

    # Update win rate
    if dashboard_state.total_trades > 0:
        dashboard_state.win_rate = (dashboard_state.winning_trades / dashboard_state.total_trades) * 100

    # Emit trade event
    socketio.emit('trade', {
        'asset': asset,
        'side': side,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'size': size,
        'pnl': pnl,
        'duration_sec': duration_sec,
        'timestamp': timestamp or datetime.now().strftime('%H:%M:%S'),
    })


def update_training_metrics(
    policy_loss: float = None,
    value_loss: float = None,
    entropy: float = None,
    kl_divergence: float = None,
    clip_fraction: float = None,
    explained_variance: float = None,
):
    """Update training metrics."""
    if policy_loss is not None:
        dashboard_state.policy_loss_history.append(policy_loss)
    if value_loss is not None:
        dashboard_state.value_loss_history.append(value_loss)
    if entropy is not None:
        dashboard_state.entropy_history.append(entropy)
    if kl_divergence is not None:
        dashboard_state.kl_divergence_history.append(kl_divergence)
    if clip_fraction is not None:
        dashboard_state.clip_fraction_history.append(clip_fraction)
    if explained_variance is not None:
        dashboard_state.explained_variance_history.append(explained_variance)

    dashboard_state.update_count += 1


def update_buffer_size(buffer_size: int, max_buffer_size: int = None):
    """Update experience buffer size."""
    dashboard_state.buffer_size = buffer_size
    if max_buffer_size is not None:
        dashboard_state.max_buffer_size = max_buffer_size


def update_episode_metrics(episode_count: int, avg_reward: float = None, avg_length: float = None):
    """Update episode-level metrics."""
    dashboard_state.episode_count = episode_count
    if avg_reward is not None:
        dashboard_state.avg_episode_reward = avg_reward
    if avg_length is not None:
        dashboard_state.avg_episode_length = avg_length


def update_value_estimates(expected_value: float, value_error: float = None):
    """Update value function estimates."""
    dashboard_state.expected_value = expected_value
    if value_error is not None:
        dashboard_state.value_error = value_error


def run_dashboard(host='0.0.0.0', port=5051):
    """Run the dashboard server."""
    import os
    port = int(os.environ.get('PORT', port))

    logger.info("=" * 60)
    logger.info("Professional RL Trading Dashboard")
    logger.info(f"{'='*60}")
    logger.info(f"URL: http://localhost:{port}")
    logger.info(f"Host: {host}")
    logger.info(f"{'='*60}\n")

    # Start metrics emitter thread
    emitter_thread = threading.Thread(target=metrics_emitter, daemon=True)
    emitter_thread.start()

    # Run Flask app
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    # Demo mode - simulate some data
    def demo_mode():
        time.sleep(2)

        # Simulate training data
        for i in range(100):
            time.sleep(0.5)

            # Simulate PnL growth
            pnl = i * 2.5 + np.random.randn() * 5
            expected_pnl = i * 2.3 + np.random.randn() * 3
            update_pnl(pnl, realized_pnl=pnl * 0.7, unrealized_pnl=pnl * 0.3)
            update_expected_pnl(expected_pnl)

            # Simulate training metrics
            if i % 5 == 0:
                update_training_metrics(
                    policy_loss=0.05 + np.random.randn() * 0.01,
                    value_loss=0.1 + np.random.randn() * 0.02,
                    entropy=1.0 - i * 0.01 + np.random.randn() * 0.05,
                    kl_divergence=0.01 + abs(np.random.randn() * 0.005),
                    clip_fraction=0.2 + np.random.randn() * 0.05,
                    explained_variance=0.5 + i * 0.005
                )

                update_buffer_size(min(256, i * 10), 256)
                update_episode_metrics(i, avg_reward=pnl / (i + 1))
                update_value_estimates(expected_pnl, value_error=abs(pnl - expected_pnl))

            # Simulate trades
            if i % 8 == 0:
                side = 'LONG' if np.random.rand() > 0.5 else 'SHORT'
                entry = 0.5 + np.random.rand() * 0.3
                exit_price = entry + (np.random.randn() * 0.05)
                trade_pnl = (exit_price - entry) * 10 if side == 'LONG' else (entry - exit_price) * 10

                log_trade(
                    asset='BTC',
                    side=side,
                    entry_price=entry,
                    exit_price=exit_price,
                    size=10.0,
                    pnl=trade_pnl,
                    duration_sec=np.random.randint(30, 300)
                )

    demo_thread = threading.Thread(target=demo_mode, daemon=True)
    demo_thread.start()

    run_dashboard()
