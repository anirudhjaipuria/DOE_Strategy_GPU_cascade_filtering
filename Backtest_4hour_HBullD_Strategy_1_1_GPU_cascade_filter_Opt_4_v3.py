import json
import os
import glob
import time
import warnings
import pickle
from typing import Dict, List, Tuple
from itertools import product, islice

import numpy as np
import pandas as pd
import cupy as cp
from multiprocessing import Pool, cpu_count
from functools import reduce
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

import psutil
import threading
import time
from pynvml import *
import numpy as np  # Already imported, but ensure it's there
from tabulate import tabulate  # For pretty-printing the matrix (pip install tabulate)

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

warnings.filterwarnings("ignore")

# ====================== DYNAMIC NAMING BASED ON SCRIPT ITSELF ======================
SCRIPT_BASE = os.path.splitext(os.path.basename(__file__))[0]
# ================================================================================

Risk_percentage = 1.00
Brokerage = 0.1  # in %
Brokerage_buy = 1 + (Brokerage / 100)
Brokerage_sell = 1 - (Brokerage / 100)

CACHE_DIR = os.path.join(os.getcwd(), ".parquet_cache_hbull_1_1")
os.makedirs(CACHE_DIR, exist_ok=True)

BLOCK_SIZE = 512                    # 128 → 256 → 512

PARAM_BATCH_SIZE = 262144           # 2048 → 4096 → 8192 → 16384 → 32768 → 65536 → 131072 → 262144 values also possible

COMBO_BATCH_SIZE = 15500000         # Adjust based on RAM; 1M should use ~1-2GB per batch

# ────────────────────────────────────────────────
# NEW: Summary cache for 30k+ parquet files
# ────────────────────────────────────────────────
SUMMARY_CACHE_DIR = os.path.join(os.getcwd(), ".summary_cache_hbull_1_1")
os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)

# Higher concurrency because we now use threads (I/O bound)
PARQUET_LOAD_PROCESSES = 96   # you can go up to 48, 64, 96, 128, 192 on a good SSD


def load_or_cache_symbol_data(folder_path: str, symbol: str) -> list:
    """Load from ultra-fast summary parquet or build it once."""
    parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
    if not parquet_files:
        print(f"    No parquet files found for {symbol}")
        return []

    mtimes = tuple(os.path.getmtime(f) for f in parquet_files)
    cache_key = abs(hash(mtimes)) % 1000000000

    summary_path = os.path.join(SUMMARY_CACHE_DIR, f"summary_{symbol}_{cache_key}.parquet")
    pkl_cache_path = os.path.join(CACHE_DIR, f"processed_data_{symbol}_{cache_key}.pkl")

    # 1. Fast path — single parquet read
    if os.path.exists(summary_path):
        df_summary = pd.read_parquet(summary_path)
        data = df_summary.to_dict('records')
        print(f"    LOADED FROM SUMMARY PARQUET → {symbol} ({len(data):,} bars)")
        return data

    # 2. Backward compatibility with old pickle
    if os.path.exists(pkl_cache_path):
        with open(pkl_cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"    LOADED FROM OLD PKL → {symbol} ({len(data):,} bars)")
        # Create summary for next runs
        pd.DataFrame(data).to_parquet(summary_path, index=False, compression='snappy')
        print(f"    CREATED SUMMARY PARQUET from old pkl")
        return data

    # 3. First time only — process all files
    print(f"    Found {len(parquet_files):,} parquet files → building summary (ONCE only)")
    data = parallel_load_files(parquet_files)

    if data:
        df_summary = pd.DataFrame(data)
        df_summary.to_parquet(summary_path, index=False, compression='snappy')
        print(f"    SAVED SUMMARY PARQUET → {symbol} ({len(data):,} bars)")

        # keep old pkl for compatibility
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(pkl_cache_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f"    WARNING: No data for {symbol}")

    return data

# ────────────────────────────────────────────────
# BACKTEST KERNEL (with floorf)
# ────────────────────────────────────────────────
BACKTEST_KERNEL_SRC = r'''

extern "C" __global__
void backtest_kernel_params_metrics(

    // Market data
    const float* opens_d,
    const float* closes_d,
    const float* prev_closes_d,
    const float* lm_lows_d,
    const long long* lm_dates_int_d,
    const long long* dates_int_d,

    // Indicator data
    const float* hb_gen_d,
    const float* hb_neg_macd_d,
    const float* hb_hl_rsi_gen_d,
    const float* hb_ll_rsi_gen_d,
    const float* hb_gen_slope_d,
    const float* hb_gen_macd_slope_d,
    const float* hb_date_gap_gen_d,
    const float* hb_hl_gen_d,

    // Parameter arrays
    const float* hb_hl_upper_d,
    const float* hb_hl_lower_d,
    const float* hb_ll_upper_d,
    const float* hb_ll_lower_d,
    const float* hb_gen_slope_upper_d,
    const float* hb_gen_slope_lower_d,
    const float* hb_gen_macd_slope_upper_d,
    const float* hb_gen_macd_slope_lower_d,
    const float* date_gap_upper_d,
    const float* date_gap_lower_d,

    // Year indexing
    const int* year_start_idx_d,
    const int* year_end_idx_d,

    // Outputs
    float* out_final_capital_d,
    float* out_max_drawdown_d,
    float* out_year_start_caps_d,
    float* out_year_end_caps_d,

    int num_params,
    int num_files,
    int num_years,
    float brokerage_buy,
    float brokerage_sell,
    float risk_percentage
)
{
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= num_params) return;

    float current_capital = 10000.0f;
    float max_capital = 10000.0f;
    float max_drawdown = 0.0f;
    float available_capital = 10000.0f;
    float total_buy_qty = 0.0f;
    float stoploss = 0.0f;
    long long first_buy_date = 0;
    int buy_signal_prev = 0;
    int stoploss_trigger_prev = 0;
    float total_buy_qty_prev = 0.0f;
    float stoploss_prev = 0.0f;

    int year_ptr = 0;
    int base_year_idx = p * num_years;

    // Load parameters once into registers
    float hl_upper = hb_hl_upper_d[p];
    float hl_lower = hb_hl_lower_d[p];
    float ll_upper = hb_ll_upper_d[p];
    float ll_lower = hb_ll_lower_d[p];
    float slope_upper = hb_gen_slope_upper_d[p];
    float slope_lower = hb_gen_slope_lower_d[p];
    float macd_slope_upper = hb_gen_macd_slope_upper_d[p];
    float macd_slope_lower = hb_gen_macd_slope_lower_d[p];
    float gap_upper = date_gap_upper_d[p];
    float gap_lower = date_gap_lower_d[p];

    for (int i = 0; i < num_files; ++i)
    {
        // Compute buy condition inline (FUSED — no 2D array)
        int buy_signal =
            ((hb_gen_d[i] == 1.0f) || 
             (hb_gen_d[i] == 1.0f && hb_neg_macd_d[i] == 1.0f)) &&
            (hb_hl_rsi_gen_d[i] < hl_upper) &&
            (hb_hl_rsi_gen_d[i] > hl_lower) &&
            (hb_ll_rsi_gen_d[i] < ll_upper) &&
            (hb_ll_rsi_gen_d[i] > ll_lower) &&
            (hb_gen_slope_d[i] < slope_upper) &&
            (hb_gen_slope_d[i] > slope_lower) &&
            (hb_gen_macd_slope_d[i] < macd_slope_upper) &&
            (hb_gen_macd_slope_d[i] > macd_slope_lower) &&
            (hb_date_gap_gen_d[i] < gap_upper) &&
            (hb_date_gap_gen_d[i] > gap_lower) &&
            (closes_d[i] > hb_hl_gen_d[i]);

        float initial_stop = buy_signal ? hb_hl_gen_d[i] : 0.0f;

        // Full backtest logic (adapted from the complete kernel)
        int actual_sell = 0;

        if (i > 0) {
            if (buy_signal_prev == 1 && stoploss_trigger_prev != 1) {
                float loss_per_unit = (brokerage_buy * prev_closes_d[i]) - (brokerage_sell * stoploss_prev);
                float buy_qty = 0.0f;
                if (loss_per_unit > 0.0f) {
                    buy_qty = (risk_percentage * available_capital) / loss_per_unit;
                }
                if (buy_qty * opens_d[i] * brokerage_buy > available_capital) {
                    buy_qty = available_capital / (opens_d[i] * brokerage_buy);
                }
                if (buy_qty < 0.0f) {
                    buy_qty = 0.0f;
                }

                available_capital = available_capital - (buy_qty * opens_d[i] * brokerage_buy);
                total_buy_qty = total_buy_qty_prev + buy_qty;
                if (buy_qty > 0.0f && total_buy_qty_prev == 0.0f) {
                    first_buy_date = dates_int_d[i];
                }
            } else if (stoploss_trigger_prev == 1 && buy_signal_prev != 1 && total_buy_qty_prev > 0.0f) {
                float proceeds = total_buy_qty_prev * opens_d[i] * brokerage_sell;
                available_capital = available_capital + proceeds;
                total_buy_qty = 0.0f;
                stoploss = 0.0f;
                actual_sell = 1;
            } else {
                total_buy_qty = total_buy_qty_prev;
            }
        } else {
            total_buy_qty = total_buy_qty_prev;
        }

        if (buy_signal == 1) {
            stoploss = initial_stop;
        } else if (total_buy_qty == 0.0f) {
            stoploss = 0.0f;
        } else {
            stoploss = stoploss_prev;
        }

        if (buy_signal != 1 && total_buy_qty > 0.0f) {
            stoploss = stoploss_prev;
        }
        if (buy_signal == 1 && actual_sell != 1 && total_buy_qty_prev > 0.0f) {
            stoploss = stoploss < stoploss_prev ? stoploss : stoploss_prev;
        }
        if (buy_signal != 1 && actual_sell != 1 && total_buy_qty_prev > 0.0f) {
            if (lm_dates_int_d[i] > first_buy_date && lm_lows_d[i] > stoploss_prev) {
                stoploss = lm_lows_d[i];
            }
        }

        int stoploss_trigger = (closes_d[i] < stoploss && total_buy_qty > 0.0f) ? 1 : 0;

        current_capital = available_capital + (total_buy_qty * closes_d[i]);

        buy_signal_prev = buy_signal;
        stoploss_trigger_prev = stoploss_trigger;
        total_buy_qty_prev = total_buy_qty;
        stoploss_prev = stoploss;

        if (year_ptr < num_years && i == year_start_idx_d[year_ptr]) {
            out_year_start_caps_d[base_year_idx + year_ptr] = current_capital;
        }
        if (year_ptr < num_years && i == year_end_idx_d[year_ptr]) {
            out_year_end_caps_d[base_year_idx + year_ptr] = current_capital;
            year_ptr += 1;
        }
        if (current_capital > max_capital) {
            max_capital = current_capital;
        } else {
            float dd = (max_capital - current_capital) / max_capital;
            if (dd > max_drawdown) {
                max_drawdown = dd;
            }
        }
    }

    out_final_capital_d[p] = current_capital;
    out_max_drawdown_d[p] = max_drawdown * 100.0f;
}
'''

BACKTEST_KERNEL = cp.RawKernel(BACKTEST_KERNEL_SRC, "backtest_kernel_params_metrics")

def process_parquet_file(file_path):
    try:
        required_columns = [
            'date', 'open', 'high', 'low', 'close', 'EMA_20', 'EMA_50', 'EMA_200',
            'HBullD_gen', 'HBullD_neg_MACD', 'HBullD_Higher_Low_RSI_gen', 'HBullD_Lower_Low_RSI_gen',
            'HBullD_Lower_Low_gen', 'HBullD_Higher_Low_RSI_neg_MACD', 'HBullD_Lower_Low_RSI_neg_MACD',
            'HBullD_Lower_Low_neg_MACD', 'HBullD_Higher_Low_gen', 'HBullD_Higher_Low_neg_MACD', 'HBullD_Date_Gap_gen',
            'HBullD_Date_Gap_neg_MACD', 'HBullD_gen_Slope', 'HBullD_neg_MACD_Slope', 'HBullD_gen_MACD_Slope',
            'HBullD_neg_MACD_MACD_Slope',
            'LM_Low_window_1_CS',
        ]

        table = pq.read_table(file_path, columns=required_columns)
        df = table.to_pandas().tail(100)

        if len(df) < 2:
            return None

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()

        lm_low, lm_date = (
            non_zero_df['LM_Low_window_1_CS'].iloc[-1],
            non_zero_df['date'].iloc[-1]
        ) if not non_zero_df.empty else (0, 0)

        return {
            'date': last_row['date'],
            'open': last_row['open'],
            'high': last_row['high'],
            'low': last_row['low'],
            'close': last_row['close'],
            'prev_close': prev_row['close'],
            'ema20_prev': prev_row['EMA_20'],
            'ema50_prev': prev_row['EMA_50'],
            'ema200_prev': prev_row['EMA_200'],
            'hb_gen': prev_row['HBullD_gen'],
            'hb_neg_macd': prev_row['HBullD_neg_MACD'],
            'hb_hl_rsi_gen': prev_row['HBullD_Higher_Low_RSI_gen'],
            'hb_ll_rsi_gen': prev_row['HBullD_Lower_Low_RSI_gen'],
            'hb_ll_gen': prev_row['HBullD_Lower_Low_gen'],
            'hb_hl_rsi_neg': prev_row['HBullD_Higher_Low_RSI_neg_MACD'],
            'hb_ll_rsi_neg': prev_row['HBullD_Lower_Low_RSI_neg_MACD'],
            'hb_ll_neg': prev_row['HBullD_Lower_Low_neg_MACD'],
            'lm_low': lm_low,
            'lm_date': lm_date,
            'hb_hl_gen': prev_row['HBullD_Higher_Low_gen'],
            'hb_hl_neg': prev_row['HBullD_Higher_Low_neg_MACD'],
            'hb_date_gap_gen': prev_row['HBullD_Date_Gap_gen'],
            'hb_date_gap_neg': prev_row['HBullD_Date_Gap_neg_MACD'],
            'hb_gen_slope': prev_row['HBullD_gen_Slope'],
            'hb_neg_macd_slope': prev_row['HBullD_neg_MACD_Slope'],
            'hb_gen_macd_slope': prev_row['HBullD_gen_MACD_Slope'],
            'hb_neg_macd_macd_slope': prev_row['HBullD_neg_MACD_MACD_Slope'],
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parallel_load_files(file_paths):
    """Faster I/O with threads (pyarrow releases GIL)."""
    if not file_paths:
        return []
    with ThreadPoolExecutor(max_workers=PARQUET_LOAD_PROCESSES) as executor:
        results = list(executor.map(process_parquet_file, file_paths))
    return [r for r in results if r is not None]

def run_backtest_for_folder_gpu_parallel(folder_path, param_sets_batch, data):
    num_files = len(data)

    dates = np.array([d['date'] for d in data], dtype='datetime64[ns]')
    dates_pd = pd.to_datetime(dates)  # Convert once for safe date difference

    years = dates_pd.year
    unique_years = np.array(sorted(set(years)))
    num_years = len(unique_years)
    year_start_idx = np.array([np.where(years == y)[0][0] for y in unique_years], dtype=np.int32)
    year_end_idx = np.array([np.where(years == y)[0][-1] for y in unique_years], dtype=np.int32)

    def to_np(key):
        return np.array([d[key] for d in data], dtype=np.float32)  # ← change this line

    opens = to_np('open')
    closes = to_np('close')
    prev_closes = to_np('prev_close')
    lm_lows = to_np('lm_low')
    lm_dates = np.array([d['lm_date'] for d in data], dtype='datetime64[ns]')
    hb_gen = to_np('hb_gen')
    hb_neg_macd = to_np('hb_neg_macd')
    hb_hl_rsi_gen = to_np('hb_hl_rsi_gen')
    hb_ll_rsi_gen = to_np('hb_ll_rsi_gen')
    hb_hl_gen = to_np('hb_hl_gen')
    hb_gen_slope = to_np('hb_gen_slope')
    hb_gen_macd_slope = to_np('hb_gen_macd_slope')
    hb_date_gap_gen = to_np('hb_date_gap_gen')

    ema20_prev  = to_np('ema20_prev')
    ema50_prev  = to_np('ema50_prev')
    ema200_prev = to_np('ema200_prev')

    lm_dates_int = lm_dates.astype('datetime64[ns]').astype(np.int64)
    dates_int = dates.astype('datetime64[ns]').astype(np.int64)

    opens_d = cp.asarray(opens, dtype=cp.float32)
    closes_d = cp.asarray(closes, dtype=cp.float32)
    prev_closes_d = cp.asarray(prev_closes, dtype=cp.float32)
    lm_lows_d = cp.asarray(lm_lows, dtype=cp.float32)
    lm_dates_int_d = cp.asarray(lm_dates_int, dtype=cp.int64)
    dates_int_d = cp.asarray(dates_int, dtype=cp.int64)
    year_start_idx_d = cp.asarray(year_start_idx, dtype=cp.int32)
    year_end_idx_d = cp.asarray(year_end_idx, dtype=cp.int32)

    hb_gen_d = cp.asarray(hb_gen, dtype=cp.float32)
    hb_neg_macd_d = cp.asarray(hb_neg_macd, dtype=cp.float32)
    hb_hl_rsi_gen_d = cp.asarray(hb_hl_rsi_gen, dtype=cp.float32)
    hb_ll_rsi_gen_d = cp.asarray(hb_ll_rsi_gen, dtype=cp.float32)
    hb_hl_gen_d = cp.asarray(hb_hl_gen, dtype=cp.float32)
    hb_gen_slope_d = cp.asarray(hb_gen_slope, dtype=cp.float32)
    hb_gen_macd_slope_d = cp.asarray(hb_gen_macd_slope, dtype=cp.float32)
    hb_date_gap_gen_d = cp.asarray(hb_date_gap_gen, dtype=cp.float32)

    ema20_prev_d  = cp.asarray(ema20_prev,  dtype=cp.float32)
    ema50_prev_d  = cp.asarray(ema50_prev,  dtype=cp.float32)
    ema200_prev_d = cp.asarray(ema200_prev, dtype=cp.float32)

    num_params_batch = len(param_sets_batch)
    results_dfs = []  # Collect sub-batch DFs

    for sub_batch_start in range(0, num_params_batch, PARAM_BATCH_SIZE):
        sub_batch_end = min(sub_batch_start + PARAM_BATCH_SIZE, num_params_batch)
        sub_batch_params = param_sets_batch[sub_batch_start:sub_batch_end]
        sub_batch_size = len(sub_batch_params)

        hb_hl_upper_d = cp.array([p['hb_hl_upper'] for p in sub_batch_params], dtype=cp.float32)
        hb_hl_lower_d = cp.array([p['hb_hl_lower'] for p in sub_batch_params], dtype=cp.float32)
        hb_ll_upper_d = cp.array([p['hb_ll_upper'] for p in sub_batch_params], dtype=cp.float32)
        hb_ll_lower_d = cp.array([p['hb_ll_lower'] for p in sub_batch_params], dtype=cp.float32)
        hb_gen_slope_upper_d = cp.array([p['hb_gen_slope_upper'] for p in sub_batch_params], dtype=cp.float32)
        hb_gen_slope_lower_d = cp.array([p['hb_gen_slope_lower'] for p in sub_batch_params], dtype=cp.float32)
        hb_gen_macd_slope_upper_d = cp.array([p['hb_gen_macd_slope_upper'] for p in sub_batch_params], dtype=cp.float32)
        hb_gen_macd_slope_lower_d = cp.array([p['hb_gen_macd_slope_lower'] for p in sub_batch_params], dtype=cp.float32)
        date_gap_upper_d = cp.array([p['date_gap_upper'] for p in sub_batch_params], dtype=cp.float32)
        date_gap_lower_d = cp.array([p['date_gap_lower'] for p in sub_batch_params], dtype=cp.float32)

        out_final_capital_d = cp.zeros(sub_batch_size, dtype=cp.float32)
        out_max_drawdown_d = cp.zeros(sub_batch_size, dtype=cp.float32)
        out_year_start_caps_d = cp.zeros(sub_batch_size * num_years, dtype=cp.float32)
        out_year_end_caps_d = cp.zeros(sub_batch_size * num_years, dtype=cp.float32)

        block = (BLOCK_SIZE, 1)
        grid = ((sub_batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)

        BACKTEST_KERNEL(grid, block, (
            opens_d, closes_d, prev_closes_d, lm_lows_d, lm_dates_int_d, dates_int_d,
        
            hb_gen_d,
            hb_neg_macd_d,
            hb_hl_rsi_gen_d,
            hb_ll_rsi_gen_d,
            hb_gen_slope_d,
            hb_gen_macd_slope_d,
            hb_date_gap_gen_d,
            hb_hl_gen_d,
        
            hb_hl_upper_d,
            hb_hl_lower_d,
            hb_ll_upper_d,
            hb_ll_lower_d,
            hb_gen_slope_upper_d,
            hb_gen_slope_lower_d,
            hb_gen_macd_slope_upper_d,
            hb_gen_macd_slope_lower_d,
            date_gap_upper_d,
            date_gap_lower_d,

            year_start_idx_d,
            year_end_idx_d,
        
            out_final_capital_d,
            out_max_drawdown_d,
            out_year_start_caps_d,
            out_year_end_caps_d,
        
            sub_batch_size,
            num_files,
            num_years,
            cp.float32(Brokerage_buy),
            cp.float32(Brokerage_sell),
            cp.float32(Risk_percentage)
        ))


        final_capitals = out_final_capital_d.get()
        max_drawdowns = out_max_drawdown_d.get()
        year_start_caps = out_year_start_caps_d.get().reshape(sub_batch_size, num_years)
        year_end_caps = out_year_end_caps_d.get().reshape(sub_batch_size, num_years)

        # Vectorized yearly ROIs (handle div-by-zero with np.where)
        yearly_rois_all = np.where(
            year_start_caps > 0,
            ((year_end_caps - year_start_caps) / year_start_caps) * 100,
            np.nan  # Use nan to skip in stats
        )

        # Vectorized stats
        average_rois = np.nanmean(yearly_rois_all, axis=1)
        yearly_medians = np.nanmedian(yearly_rois_all, axis=1)
        yearly_stds = np.nanstd(yearly_rois_all, axis=1, ddof=0)
        yearly_cvs = np.where(average_rois != 0, (yearly_stds / average_rois) * 100, 0.0)

        # Vectorized total CAGR
        start_value_total = 10000.0
        end_value_total = final_capitals
        first_date_pd = dates_pd[0]
        last_date_pd = dates_pd[-1]
        delta = last_date_pd - first_date_pd
        num_years_total = delta.days / 365.25 if delta.days > 0 else 0.0
        cagrs = np.where(
            (num_years_total > 0) & (start_value_total > 0),
            ((end_value_total / start_value_total) ** (1 / num_years_total) - 1) * 100,
            0.0
        )

        # Build DataFrame directly from params + results (no list of dicts)
        results_df = pd.DataFrame(sub_batch_params)  # Columns from param keys
        results_df['Final_Capital'] = final_capitals
        results_df['Average_ROI'] = average_rois
        results_df['CAGR'] = cagrs
        results_df['Max_Drawdown'] = max_drawdowns
        results_df['yearly_CAGR_mean'] = average_rois  # Same as Average_ROI, but keep for consistency
        results_df['yearly_CAGR_median'] = yearly_medians
        results_df['yearly_CAGR_std'] = yearly_stds
        results_df['yearly_CAGR_cv'] = yearly_cvs

        # Add per-year CAGRs (comment out if not needed to save memory)
        for j, year in enumerate(unique_years):
            results_df[f'CAGR_{year}'] = yearly_rois_all[:, j]

        results_dfs.append(results_df)

    # Concat all sub-batch DFs
    if results_dfs:
        return pd.concat(results_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def run_auswertung(symbols, input_suffix):
    file_paths = [
        f'full_optimization_results_{symbol}_{SCRIPT_BASE}_{input_suffix}.parquet'
        for symbol in symbols
    ]

    currencies = symbols.copy()

    parameter_cols = [
        'hb_hl_upper', 'hb_hl_lower', 'hb_ll_upper', 'hb_ll_lower',
        'hb_gen_slope_upper', 'hb_gen_slope_lower', 'hb_gen_macd_slope_upper',
        'hb_gen_macd_slope_lower', 'date_gap_upper', 'date_gap_lower'
    ]

    dfs = []
    for file_path, symbol in zip(file_paths, currencies):
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        df = pd.read_parquet(file_path)
        metric_cols = [col for col in df.columns if col not in parameter_cols]
        rename_dict = {col: f"{col}_{symbol}" for col in metric_cols}
        df = df.rename(columns=rename_dict)
        dfs.append(df)

    if not dfs:
        print("No files were read. Exiting.")
        return

    print(f"Merging {len(dfs)} dataframes...")
    merged = reduce(
        lambda left, right: pd.merge(left, right, on=parameter_cols, how='inner'),
        dfs
    )

    # ────────────────────────────────────────────────
    # Filter: all overall CAGRs > 0
    # ────────────────────────────────────────────────
    overall_cagr_cols = [f'CAGR_{symbol}' for symbol in currencies]
    print(f"Filtering for positive overall CAGR across all symbols...")
    mask = (merged[overall_cagr_cols] > 0).all(axis=1)
    filtered_df = merged[mask].copy().reset_index(drop=True)

    # ────────────────────────────────────────────────
    # Calculate statistics across currencies
    # ────────────────────────────────────────────────
    print("Calculating cross-symbol statistics...")

    overall_cagr_values = filtered_df[overall_cagr_cols].values
    filtered_df['CAGR_mean'] = np.nanmean(overall_cagr_values, axis=1)
    filtered_df['CAGR_median'] = np.nanmedian(overall_cagr_values, axis=1)
    filtered_df['CAGR_std'] = np.nanstd(overall_cagr_values, axis=1, ddof=0)
    filtered_df['CAGR_cv'] = np.where(filtered_df['CAGR_mean'] != 0,
                                      (filtered_df['CAGR_std'] * 100 / filtered_df['CAGR_mean']), 0)

    maxdd_cols = [col for col in filtered_df.columns if 'Max_Drawdown' in col]
    if maxdd_cols:
        maxdd_values = filtered_df[maxdd_cols].values
        filtered_df['Max_Drawdown_mean'] = np.nanmean(maxdd_values, axis=1)

    yearly_cagr_cols = [
        col for col in filtered_df.columns
        if col.startswith('CAGR_') and col.count('_') == 2 and col.split('_')[1].isdigit()
    ]
    if yearly_cagr_cols:
        yearly_cagr_df = filtered_df[yearly_cagr_cols]
        filtered_df['pooled_yearly_CAGR_mean'] = yearly_cagr_df.mean(axis=1, skipna=True)
        filtered_df['pooled_yearly_CAGR_median'] = yearly_cagr_df.median(axis=1, skipna=True)
        filtered_df['pooled_yearly_CAGR_std'] = yearly_cagr_df.std(axis=1, skipna=True, ddof=0)
        filtered_df['pooled_yearly_CAGR_cv'] = np.where(
            filtered_df['pooled_yearly_CAGR_mean'] != 0,
            (filtered_df['pooled_yearly_CAGR_std'] / filtered_df['pooled_yearly_CAGR_mean']) * 100,
            0
        )

    # ────────────────────────────────────────────────
    # Reorder columns
    # ────────────────────────────────────────────────
    ordered_cols = []
    for metric in ['Final_Capital', 'CAGR', 'Average_ROI', 'Max_Drawdown']:
        metric_group = [f"{metric}_{symbol}" for symbol in currencies
                        if f"{metric}_{symbol}" in filtered_df.columns]
        ordered_cols.extend(metric_group)

    yearly_stat_metrics = ['yearly_CAGR_mean', 'yearly_CAGR_median', 'yearly_CAGR_std', 'yearly_CAGR_cv']
    for metric in yearly_stat_metrics:
        metric_group = [f"{metric}_{symbol}" for symbol in currencies
                        if f"{metric}_{symbol}" in filtered_df.columns]
        ordered_cols.extend(metric_group)

    yearly_cagr_cols = sorted([
        col for col in filtered_df.columns
        if col.startswith('CAGR_') and col.count('_') == 2 and col.split('_')[1].isdigit()
    ])
    ordered_cols.extend(yearly_cagr_cols)

    summary_cols = [
        'CAGR_mean', 'CAGR_median', 'CAGR_std', 'CAGR_cv',
        'pooled_yearly_CAGR_mean', 'pooled_yearly_CAGR_median',
        'pooled_yearly_CAGR_std', 'pooled_yearly_CAGR_cv',
        'Max_Drawdown_mean'
    ]
    ordered_cols.extend([col for col in summary_cols if col in filtered_df.columns])
    ordered_cols.extend(parameter_cols)

    filtered_df = filtered_df[ordered_cols]

    # ────────────────────────────────────────────────
    # Round all numeric columns to 3 decimal places
    # ────────────────────────────────────────────────
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    filtered_df[numeric_cols] = filtered_df[numeric_cols].round(3)

    # ====================== FINAL STATS FILE ======================
    output_file = f'{SCRIPT_BASE}_with_stats.parquet'
    filtered_df.to_parquet(output_file, index=False)

    print(f"\n✅ Done!")
    print(f"📁 Final stats saved as → {output_file}")
    print(f"Number of remaining combinations: {len(filtered_df):,}")
    print(f"Symbols included: {len(overall_cagr_cols)}")
    if len(filtered_df) > 0:
        print(f"Index range: {filtered_df.index.min()} – {filtered_df.index.max()}")
        print("\nNew/updated columns added:")
        print("  - CAGR_mean, CAGR_median, CAGR_std, CAGR_cv")
        print("  - pooled_yearly_CAGR_mean, pooled_yearly_CAGR_median, ...")
        print("  - Max_Drawdown_mean")
        print("\nAll numeric values rounded to 3 decimal places.")
    else:
        print("Warning: No combinations survived the filter.")

if __name__ == '__main__':
    # Initialize monitoring data structures
    monitoring_data = {
        'cpu_percent': [],
        'ram_percent': [],
        'gpu_util': [],  # GPU core utilization % (SM occupancy, correlates to core usage)
        'gpu_vram_used': [],  # In MB
        'gpu_vram_total': 0  # Set once
    }

    # Initialize NVML for GPU
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU (RTX 5080); change index if multi-GPU
    info = nvmlDeviceGetMemoryInfo(handle)
    monitoring_data['gpu_vram_total'] = info.total // (1024 * 1024)  # MB


    # Monitoring thread function
    def monitor_resources(stop_event):
        while not stop_event.is_set():
            # CPU
            monitoring_data['cpu_percent'].append(psutil.cpu_percent(interval=0.1))

            # RAM
            monitoring_data['ram_percent'].append(psutil.virtual_memory().percent)

            # GPU Utilization (cores %)
            util = nvmlDeviceGetUtilizationRates(handle)
            monitoring_data['gpu_util'].append(util.gpu)  # GPU core utilization %

            # GPU VRAM
            info = nvmlDeviceGetMemoryInfo(handle)
            monitoring_data['gpu_vram_used'].append(info.used // (1024 * 1024))  # MB

            time.sleep(5)  # Sample every 5 seconds; adjust for granularity (e.g., 1 for more data)


    # Start monitoring thread
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring,))
    monitor_thread.start()

    start_time = time.time()

    # ────────────────────────────────────────────────
    # FIND ALL JSON FILES
    # ────────────────────────────────────────────────
    json_files = sorted(glob.glob("doe_params_HBullD_Strategy_*.json"))
    if not json_files:
        print("No JSON files found matching 'doe_params_HBullD_Strategy_*.json'")
        exit(1)
    print(f"Found {len(json_files)} JSON files to process")

    suffixes = [
        os.path.basename(json_path).split('Strategy_')[-1].replace('.json', '')
        for json_path in json_files
    ]

    # ────────────────────────────────────────────────
    # SYMBOLS TO PROCESS
    # ────────────────────────────────────────────────

    SYMBOLS = ['XLM', 'TRX', 'DOGE', 'BTC', 'BNB', 'ADA', 'XRP', 'SOL', 'ETH']

    # SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD', 'BANKINDIA', 'DABUR',
    #            'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
    #            'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN']

    base_path = r'C:\PYTHON\historical_data\CRYPTO_year'

    folders = [
        os.path.join(base_path, symbol, 'output_4hour_parquet')
        for symbol in SYMBOLS
    ]

    symbols = SYMBOLS

    print(f"Will process {len(symbols)} symbols:")
    print(", ".join(symbols))
    print(f"Total symbols: {len(symbols)}\n")

    for json_idx, json_path in enumerate(json_files, start=1):
        suffix = os.path.basename(json_path).split('Strategy_')[-1].replace('.json', '')
        print(f"\n{'='*80}")
        print(f"Processing JSON file {json_idx}/{len(json_files)}: {json_path} → suffix _{suffix}")
        print(f"{'='*80}\n")

        # ────────────────────────────────────────────────
        # LOAD PARAMETERS FROM JSON
        # ────────────────────────────────────────────────
        try:
            with open(json_path, 'r') as f:
                param_specs = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found: {json_path}")
            continue
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {json_path}")
            continue

        # Generate ranges
        param_keys = list(param_specs.keys())
        ranges = []
        for spec in param_specs.values():
            if 'value' in spec:
                ranges.append([spec['value']])
            else:
                ranges.append(list(range(spec['min'], spec['max'] + 1, spec['step'])))

        total_combos = reduce(lambda x, y: x * len(y), ranges, 1)
        print(f"Total parameter combinations before filtering: {total_combos:,}")

        # Compute total valid combinations
        def count_valid(lower_range, upper_range):
            return sum(1 for l in lower_range for u in upper_range if l < u)

        num_valid_hl = count_valid(ranges[1], ranges[0])  # hb_hl_lower, hb_hl_upper
        num_valid_ll = count_valid(ranges[3], ranges[2])  # hb_ll_lower, hb_ll_upper
        num_valid_gen_slope = count_valid(ranges[5], ranges[4])
        num_valid_gen_macd_slope = count_valid(ranges[7], ranges[6])
        num_valid_date_gap = count_valid(ranges[9], ranges[8])

        total_valid = num_valid_hl * num_valid_ll * num_valid_gen_slope * num_valid_gen_macd_slope * num_valid_date_gap
        print(f"Total valid parameter combinations: {total_valid:,}")

        if total_valid == 0:
            print("⚠️  No valid parameter combinations → skipping all symbols")

            zero_rows = []
            for asset_name in symbols:
                zero_dict = {k: 0 for k in param_keys}
                zero_result = {
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    'yearly_CAGR_mean': 0.0,
                    'yearly_CAGR_median': 0.0,
                    'yearly_CAGR_std': 0.0,
                    'yearly_CAGR_cv': 0.0,
                    'params': zero_dict
                }

                parquet_path = f"full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet"
                if not zero_rows:
                    print("No zero_rows; skipping DataFrame creation")
                    continue  # Or create empty DF: pd.DataFrame(columns=expected_cols)
                df_zero = pd.DataFrame([zero_rows[-1]])
                df_zero.to_parquet(parquet_path, index=False, compression='snappy')

                print(f"  → Wrote empty result for {asset_name}")

            # IMPORTANT: skip the rest of this JSON completely
            continue

        # Generate valid combos once
        print("Generating valid parameter combinations...")
        
        def generate_valid_combos():
            for combo in product(*ranges):
                hb_hl_upper, hb_hl_lower, hb_ll_upper, hb_ll_lower, \
                hb_gen_slope_upper, hb_gen_slope_lower, hb_gen_macd_slope_upper, \
                hb_gen_macd_slope_lower, date_gap_upper, date_gap_lower = combo
        
                if (hb_hl_lower < hb_hl_upper and
                    hb_ll_lower < hb_ll_upper and
                    hb_gen_slope_lower < hb_gen_slope_upper and
                    hb_gen_macd_slope_lower < hb_gen_macd_slope_upper and
                    date_gap_lower < date_gap_upper):
                    yield combo

        valid_combos = list(generate_valid_combos())
        print(f"Generated {len(valid_combos):,} valid combinations")

        for folder_idx, folder_path in enumerate(folders):
            asset_name = symbols[folder_idx]
            print(f"  → Processing {asset_name:12}  ({folder_idx + 1:2d}/{len(symbols)})")

            if len(valid_combos) == 0:
                print("    No surviving combinations from previous symbols → skipping backtest and saving zero result")
                # Directly create and save zero df_positive (copy the else block code here)
                zero_dict = {k: 0 for k in param_keys}
                zero_result = {
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    'yearly_CAGR_mean': 0.0,
                    'yearly_CAGR_median': 0.0,
                    'yearly_CAGR_std': 0.0,
                    'yearly_CAGR_cv': 0.0,
                    'params': zero_dict
                }
                df_positive = pd.json_normalize([zero_result])
                df_positive = df_positive.rename(columns={f"params.{k}": k for k in param_keys})
                parquet_path = f"full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet"
                df_positive.to_parquet(parquet_path, index=False, compression='snappy')
                print(f"    Saved zero result for {asset_name}")
                # No update to good_tuples/valid_combos needed, as it's already 0
                continue

            # ────────────────────────────────────────────────
            # ONLY REACHED IF valid_combos STILL EXISTS
            # ────────────────────────────────────────────────
            data = load_or_cache_symbol_data(folder_path, asset_name)
            if not data:
                print("    No valid data loaded → skipping")
                continue

            print(f"    Using {len(data):,} bars for backtesting")

            # ────────────────────────────────────────────────
            # PROCESS IN COMBO BATCHES TO SAVE RAM
            # ────────────────────────────────────────────────
            all_good_dfs = []  # Reset for each asset
            batch_num = 1
            
            for batch_start in range(0, len(valid_combos), COMBO_BATCH_SIZE):
                print(f"    Generating combo batch {batch_num}...")
                batch_combos = valid_combos[batch_start:batch_start + COMBO_BATCH_SIZE]

                param_sets_batch = [dict(zip(param_keys, combo)) for combo in batch_combos]
                print(f"    Batch size: {len(param_sets_batch):,}")

                results_df = run_backtest_for_folder_gpu_parallel(folder_path, param_sets_batch, data)
                good_df = results_df[results_df['CAGR'] > 0]
                all_good_dfs.append(good_df)

                print(f"    Processed batch {batch_num} ({len(results_df):,} results, {len(good_df):,} positive)")
                batch_num += 1

            if all_good_dfs:
                df_positive = pd.concat(all_good_dfs, ignore_index=True)

                if df_positive.empty:
                    print("\nNo positive CAGR results for this asset.")
                    zero_dict = {k: 0 for k in param_keys}
                    zero_result = {
                        'Final_Capital': 0.0,
                        'Average_ROI': 0.0,
                        'CAGR': 0.0,
                        'Max_Drawdown': 0.0,
                        'yearly_CAGR_mean': 0.0,
                        'yearly_CAGR_median': 0.0,
                        'yearly_CAGR_std': 0.0,
                        'yearly_CAGR_cv': 0.0,
                        'params': zero_dict
                    }
                    df_positive = pd.json_normalize([zero_result])
                    df_positive = df_positive.rename(columns={f"params.{k}": k for k in param_keys})

                # Save the results (positive or zero) — using script name
                parquet_path = f"full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet"
                df_positive.to_parquet(parquet_path, index=False, compression='snappy')
                print(f"    Saved → {parquet_path} ({len(df_positive):,} rows)")

                good_tuples = set(map(tuple, df_positive[param_keys].values))

            else:
                print("\nNo positive CAGR results for this asset.")
                zero_dict = {k: 0 for k in param_keys}
                zero_result = {
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    'yearly_CAGR_mean': 0.0,
                    'yearly_CAGR_median': 0.0,
                    'yearly_CAGR_std': 0.0,
                    'yearly_CAGR_cv': 0.0,
                    'params': zero_dict
                }
                df_positive = pd.json_normalize([zero_result])
                df_positive = df_positive.rename(columns={f"params.{k}": k for k in param_keys})
                parquet_path = f"full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet"
                df_positive.to_parquet(parquet_path, index=False, compression='snappy')
                good_tuples = set()

            # Filter valid_combos for next symbol
            valid_combos = [c for c in valid_combos if c in good_tuples]
            print(f"    Surviving combinations for next symbol: {len(valid_combos):,}")

    # ====================== COMBINE + CLEANUP ======================
    parameter_cols = [
        'hb_hl_upper', 'hb_hl_lower', 'hb_ll_upper', 'hb_ll_lower',
        'hb_gen_slope_upper', 'hb_gen_slope_lower', 'hb_gen_macd_slope_upper',
        'hb_gen_macd_slope_lower', 'date_gap_upper', 'date_gap_lower'
    ]

    print("\n" + "=" * 80)
    print("Combining results across all suffixes per symbol + cleanup...")
    print("=" * 80)

    for asset_name in symbols:
        all_dfs = []
        for suffix in suffixes:
            file_path = f'full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet'
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                all_dfs.append(df)
                print(f"  Included: {file_path}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=parameter_cols, keep='first')

            combined_path = f'full_optimization_results_{asset_name}_{SCRIPT_BASE}_combined.parquet'
            combined_df.to_parquet(combined_path, index=False, compression='snappy')
            print(f"  → Saved combined: {combined_path} ({len(combined_df):,} rows)")

            # Delete all individual per-suffix files
            for suffix in suffixes:
                indiv_path = f'full_optimization_results_{asset_name}_{SCRIPT_BASE}_{suffix}.parquet'
                if os.path.exists(indiv_path):
                    try:
                        os.remove(indiv_path)
                        print(f"    → Deleted temporary: {indiv_path}")
                    except Exception as e:
                        print(f"    Warning: Could not delete {indiv_path}: {e}")
        else:
            print(f"  No results found for {asset_name}")

    # ────────────────────────────────────────────────
    # FINAL AUSWERTUNG ON COMBINED DATA
    # ────────────────────────────────────────────────
    print("\nRunning final cross-symbol evaluation on combined results...")
    run_auswertung(symbols, "combined")

    stop_monitoring.set()
    monitor_thread.join()

    # Shutdown NVML
    nvmlShutdown()

    # Compute stats (ignore empty lists if script is too short)
    if monitoring_data['cpu_percent']:  # Only if samples were taken
        stats = {
            'CPU Load (%)': {
                'Average': np.mean(monitoring_data['cpu_percent']),
                'Peak': np.max(monitoring_data['cpu_percent']),
                'Notes': f"i9 Ultra 285K has ~24-32 cores (depending on config); high load here indicates CPU bottleneck in data loading/parameter gen."
            },
            'RAM Usage (%)': {
                'Average': np.mean(monitoring_data['ram_percent']),
                'Peak': np.max(monitoring_data['ram_percent']),
                'Notes': f"64GB DDR5; high usage (>80%) could bottleneck if swapping occurs. Script uses large arrays (e.g., COMBO_BATCH_SIZE)."
            },
            'GPU Core Utilization (%)': {
                'Average': np.mean(monitoring_data['gpu_util']),
                'Peak': np.max(monitoring_data['gpu_util']),
                'Notes': f"RTX 5080 has ~10,000+ CUDA cores; utilization >90% means GPU-bound (kernels like PRECOMPUTE_KERNEL)."
            },
            'GPU VRAM Usage (MB)': {
                'Average': np.mean(monitoring_data['gpu_vram_used']),
                'Peak': np.max(monitoring_data['gpu_vram_used']),
                'Notes': f"Total 16GB (16384 MB); high usage could cause OOM errors. Script loads large arrays to GPU (e.g., num_files * num_params)."
            }
        }

        # Identify potential bottleneck (heuristic: highest average load resource)
        loads = {k: stats[k]['Average'] for k in stats if '%' in k}
        max_load_resource = max(loads, key=loads.get)
        print(f"\nPotential Bottleneck: {max_load_resource} (highest average load). Check notes below.")

        # Print comparison matrix as table
        table_data = []
        for resource, data in stats.items():
            table_data.append([
                resource,
                f"{data['Average']:.2f}",
                f"{data['Peak']:.2f}",
                data['Notes']
            ])

        headers = ["Resource", "Average Load", "Peak Load", "Bottleneck Potential/Notes"]
        print("\nResource Usage Comparison Matrix:")
        print(tabulate(table_data, headers, tablefmt="grid", floatfmt=".2f"))
    else:
        print("\nNo monitoring data collected (script too short).")

    execution_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total execution time: {execution_time:.2f} seconds  ({execution_time/60:.1f} minutes)")
    print(f"{'='*80}")