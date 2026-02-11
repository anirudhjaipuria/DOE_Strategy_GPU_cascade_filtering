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

# Set pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

warnings.filterwarnings("ignore")

Risk_percentage = 1.00
Brokerage = 0.1  # in %
Brokerage_buy = 1 + (Brokerage / 100)
Brokerage_sell = 1 - (Brokerage / 100)

CACHE_DIR = os.path.join(os.getcwd(), ".parquet_cache_hbull_1_1")
os.makedirs(CACHE_DIR, exist_ok=True)
USE_CACHE = True

USE_GPU_PARALLEL = True
PARAM_BATCH_SIZE = 65536             # 2048 → 4096 → 8192 → 16384 → 32768 → 65536 values also possible
PRECOMPUTE_THREADS_2D = (32, 8)
KERNEL_STREAMS = 2
PARQUET_LOAD_PARALLEL = True
PARQUET_LOAD_PROCESSES = max(1, cpu_count() - 2)
PARQUET_LOAD_CHUNKSIZE = 20

ENABLE_PROFILING = True
PROFILE_BATCH_TIMES = False
TOP_PERCENT = 3.0
TOP_PERCENT_METRIC = "Final_Capital"

COMBO_BATCH_SIZE = 15000000  # Adjust based on RAM; 1M should use ~1-2GB per batch

# ────────────────────────────────────────────────
# PRECOMPUTE KERNEL (with trend enabled)
# ────────────────────────────────────────────────
PRECOMPUTE_KERNEL_SRC = r'''

extern "C" __global__
void precompute_buy_signals(
    const float* hb_gen_d,
    const float* hb_neg_macd_d,
    const float* hb_hl_rsi_gen_d,
    const float* hb_ll_rsi_gen_d,
    const float* hb_gen_slope_d,
    const float* hb_gen_macd_slope_d,
    const float* hb_date_gap_gen_d,
    const float* closes_d,
    const float* hb_hl_gen_d,
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
    const float* ema20_prev_d,
    const float* ema50_prev_d,
    const float* ema200_prev_d,
    int* buy_signals_pre_d,
    float* initial_stoploss_pre_d,
    int num_params,
    int num_files
) {
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (p >= num_params || i >= num_files) return;

    int idx = p * num_files + i;

    int hb_gen = (hb_gen_d[i] == 1.0f);
    int hb_neg = (hb_neg_macd_d[i] == 1.0f);

    // int trend_ok = (ema20_prev_d[i] > ema50_prev_d[i]) && (ema50_prev_d[i] > ema200_prev_d[i]);
    // int trend_ok = (ema50_prev_d[i] > ema200_prev_d[i]);

    int cond =
        // trend_ok &&
        ((hb_gen && hb_neg) || hb_gen) &&
        (hb_hl_rsi_gen_d[i] < hb_hl_upper_d[p]) && (hb_hl_rsi_gen_d[i] > hb_hl_lower_d[p]) &&
        (hb_ll_rsi_gen_d[i] < hb_ll_upper_d[p]) && (hb_ll_rsi_gen_d[i] > hb_ll_lower_d[p]) &&
        (hb_gen_slope_d[i] < hb_gen_slope_upper_d[p]) && (hb_gen_slope_d[i] > hb_gen_slope_lower_d[p]) &&
        (hb_gen_macd_slope_d[i] < hb_gen_macd_slope_upper_d[p]) && (hb_gen_macd_slope_d[i] > hb_gen_macd_slope_lower_d[p]) &&
        (hb_date_gap_gen_d[i] < date_gap_upper_d[p]) && (hb_date_gap_gen_d[i] > date_gap_lower_d[p]) &&
        (closes_d[i] > hb_hl_gen_d[i]);

    if (cond) {
        buy_signals_pre_d[idx] = 1;
        initial_stoploss_pre_d[idx] = hb_hl_gen_d[i];
    } else {
        buy_signals_pre_d[idx] = 0;
        initial_stoploss_pre_d[idx] = 0.0f;
    }
}
'''

# ────────────────────────────────────────────────
# BACKTEST KERNEL (with floorf)
# ────────────────────────────────────────────────
BACKTEST_KERNEL_SRC = r'''

extern "C" __global__
void backtest_kernel_params_metrics(
    const float* opens_d,
    const float* closes_d,
    const float* prev_closes_d,
    const float* lm_lows_d,
    const long long* lm_dates_int_d,
    const long long* dates_int_d,
    const int* buy_signals_pre_d,
    const float* initial_stoploss_pre_d,
    const int* year_start_idx_d,
    const int* year_end_idx_d,
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
) {
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
    int base_idx = p * num_files;
    int base_year_idx = p * num_years;

    for (int i = 0; i < num_files; ++i) {
        int idx = base_idx + i;
        int buy_signal = buy_signals_pre_d[idx] == 1 ? 1 : 0;
        float initial_stop = buy_signal == 1 ? initial_stoploss_pre_d[idx] : 0.0f;
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

                buy_qty = floorf(buy_qty);                 // integer quantity (float32 version)

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

PRECOMPUTE_KERNEL = cp.RawKernel(PRECOMPUTE_KERNEL_SRC, "precompute_buy_signals")
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

        df = pd.read_parquet(file_path, columns=required_columns).tail(100)
        if len(df) < 2:
            return None
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        non_zero_df = df[df['LM_Low_window_1_CS'] != 0][['LM_Low_window_1_CS', 'date']].dropna()
        lm_low, lm_date = (non_zero_df['LM_Low_window_1_CS'].iloc[-1], non_zero_df['date'].iloc[-1]) if not non_zero_df.empty else (0, 0)
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
    with Pool(processes=PARQUET_LOAD_PROCESSES) as pool:
        results = pool.map(process_parquet_file, file_paths)
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
    results_batch = []

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

        buy_signals_pre_d = cp.zeros(sub_batch_size * num_files, dtype=cp.int32)
        initial_stoploss_pre_d = cp.zeros(sub_batch_size * num_files, dtype=cp.float32)

        grid = ((sub_batch_size + PRECOMPUTE_THREADS_2D[0] - 1) // PRECOMPUTE_THREADS_2D[0],
                (num_files + PRECOMPUTE_THREADS_2D[1] - 1) // PRECOMPUTE_THREADS_2D[1])
        block = PRECOMPUTE_THREADS_2D
        PRECOMPUTE_KERNEL(grid, block, (
            hb_gen_d, hb_neg_macd_d, hb_hl_rsi_gen_d, hb_ll_rsi_gen_d, hb_gen_slope_d,
            hb_gen_macd_slope_d, hb_date_gap_gen_d, closes_d, hb_hl_gen_d,
            hb_hl_upper_d, hb_hl_lower_d, hb_ll_upper_d, hb_ll_lower_d,
            hb_gen_slope_upper_d, hb_gen_slope_lower_d,
            hb_gen_macd_slope_upper_d, hb_gen_macd_slope_lower_d,
            date_gap_upper_d, date_gap_lower_d,
            ema20_prev_d, ema50_prev_d, ema200_prev_d,
            buy_signals_pre_d, initial_stoploss_pre_d,
            sub_batch_size, num_files
        ))

        out_final_capital_d = cp.zeros(sub_batch_size, dtype=cp.float32)
        out_max_drawdown_d = cp.zeros(sub_batch_size, dtype=cp.float32)
        out_year_start_caps_d = cp.zeros(sub_batch_size * num_years, dtype=cp.float32)
        out_year_end_caps_d = cp.zeros(sub_batch_size * num_years, dtype=cp.float32)

        grid = ((sub_batch_size + 31) // 32, 1)
        block = (32, 1)
        BACKTEST_KERNEL(grid, block, (
            opens_d, closes_d, prev_closes_d, lm_lows_d, lm_dates_int_d, dates_int_d,
            buy_signals_pre_d, initial_stoploss_pre_d,
            year_start_idx_d, year_end_idx_d,
            out_final_capital_d, out_max_drawdown_d,
            out_year_start_caps_d, out_year_end_caps_d,
            sub_batch_size, num_files, num_years,
            cp.float32(Brokerage_buy),
            cp.float32(Brokerage_sell),
            cp.float32(Risk_percentage)
        ))

        final_capitals = out_final_capital_d.get()
        max_drawdowns = out_max_drawdown_d.get()
        year_start_caps = out_year_start_caps_d.get().reshape(sub_batch_size, num_years)
        year_end_caps = out_year_end_caps_d.get().reshape(sub_batch_size, num_years)

        for i in range(sub_batch_size):
            yearly_rois = ((year_end_caps[i] - year_start_caps[i]) / year_start_caps[i]) * 100
            average_roi = np.mean(yearly_rois) if len(yearly_rois) > 0 else 0.0

            start_value_total = 10000.0
            end_value_total = final_capitals[i]

            first_date_pd = dates_pd[0]
            last_date_pd = dates_pd[-1]
            delta = last_date_pd - first_date_pd
            num_years_total = delta.days / 365.25 if delta.days > 0 else 0.0

            cagr = 0.0
            if num_years_total > 0 and start_value_total > 0:
                cagr = ((end_value_total / start_value_total) ** (1 / num_years_total) - 1) * 100

            results_batch.append({
                'Final_Capital': final_capitals[i],
                'Average_ROI': average_roi,
                'CAGR': cagr,
                'Max_Drawdown': max_drawdowns[i],
                'params': sub_batch_params[i]
            })

    return results_batch


def run_auswertung(symbols, suffix):
    file_paths = [
        f'full_optimization_results_{symbol}_HBullD_{suffix}.parquet'
        for symbol in symbols
    ]

    currencies = symbols.copy()

    parameter_cols = [
        'hb_hl_upper', 'hb_hl_lower', 'hb_ll_upper', 'hb_ll_lower',
        'hb_gen_slope_upper', 'hb_gen_slope_lower', 'hb_gen_macd_slope_upper',
        'hb_gen_macd_slope_lower', 'date_gap_upper', 'date_gap_lower'
    ]

    metric_names = ['Final_Capital', 'CAGR', 'Average_ROI', 'Max_Drawdown']

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

    # ────────────────────────────────────────────────
    # Merge all dataframes
    # ────────────────────────────────────────────────
    print(f"Merging {len(dfs)} dataframes...")
    merged = reduce(
        lambda left, right: pd.merge(left, right, on=parameter_cols, how='inner'),
        dfs
    )

    # ────────────────────────────────────────────────
    # Filter: all CAGRs > 0
    # ────────────────────────────────────────────────
    cagr_cols = [col for col in merged.columns if 'CAGR' in col]
    print(f"Found {len(cagr_cols)} CAGR columns.")

    print("Filtering for positive CAGR across all symbols...")
    mask = (merged[cagr_cols] > 0).all(axis=1)
    filtered_df = merged[mask].copy()

    # Reset index
    filtered_df = filtered_df.reset_index(drop=True)

    # ────────────────────────────────────────────────
    # Calculate statistics across currencies
    # ────────────────────────────────────────────────
    print("Calculating cross-symbol statistics...")

    # CAGR statistics
    cagr_values = filtered_df[cagr_cols].values
    filtered_df['CAGR_mean'] = np.nanmean(cagr_values, axis=1)
    filtered_df['CAGR_median'] = np.nanmedian(cagr_values, axis=1)
    filtered_df['CAGR_std'] = np.nanstd(cagr_values, axis=1, ddof=0)
    filtered_df['CAGR_sigma'] = filtered_df['CAGR_std']  # same value, just clearer name

    # Max Drawdown mean
    maxdd_cols = [col for col in filtered_df.columns if 'Max_Drawdown' in col]
    if maxdd_cols:
        maxdd_values = filtered_df[maxdd_cols].values
        filtered_df['Max_Drawdown_mean'] = np.nanmean(maxdd_values, axis=1)

    # ────────────────────────────────────────────────
    # Reorder columns: metrics grouped, then stats, then parameters
    # ────────────────────────────────────────────────
    ordered_cols = []
    for metric in metric_names:
        metric_group = [f"{metric}_{symbol}" for symbol in currencies
                        if f"{metric}_{symbol}" in filtered_df.columns]
        ordered_cols.extend(metric_group)

    # Summary statistics columns
    summary_cols = [
        'CAGR_mean', 'CAGR_median', 'CAGR_std', 'CAGR_sigma',
        'Max_Drawdown_mean'
    ]
    ordered_cols.extend([col for col in summary_cols if col in filtered_df.columns])

    # Finally the parameters
    ordered_cols.extend(parameter_cols)

    # Apply column order
    filtered_df = filtered_df[ordered_cols]

    # ────────────────────────────────────────────────
    # Round all numeric columns to 3 decimal places
    # ────────────────────────────────────────────────
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    filtered_df[numeric_cols] = filtered_df[numeric_cols].round(3)

    # ────────────────────────────────────────────────
    # Save
    # ────────────────────────────────────────────────
    output_file = f'filtered_positive_cagr_with_stats_{suffix}.parquet'
    filtered_df.to_parquet(output_file, index=False)

    print(f"\nDone!")
    print(f"Saved to: {output_file}")
    print(f"Number of remaining combinations: {len(filtered_df):,}")
    print(f"Symbols included: {len(cagr_cols)}")
    if len(filtered_df) > 0:
        print(f"Index range: {filtered_df.index.min()} – {filtered_df.index.max()}")
        print("\nNew/updated columns added:")
        print("  - CAGR_mean")
        print("  - CAGR_median")
        print("  - CAGR_std")
        print("  - CAGR_sigma     ← standard deviation expressed as sigma")
        print("  - Max_Drawdown_mean")
        print("\nAll numeric values rounded to 3 decimal places.")
    else:
        print("Warning: No combinations survived the filter.")

if __name__ == '__main__':
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

    SYMBOLS = ['DOT', 'LINK', 'ETH', 'BNB', 'BTC', 'DOGE', 'ADA', 'XRP', 'SOL', 'TRX', 'XLM']

    # SYMBOLS = ['ADANIENT', 'AXISBANK', 'BAJFINANC', 'BANKBAROD', 'BANKINDIA', 'DABUR',
    #            'HDFCBANK', 'HINDUNILV', 'ICICIBANK', 'INDUSINDB', 'INFY', 'ITC',
    #            'KOTAKBANK', 'LT', 'NOCIL', 'RAJESHEXP', 'SBIN', 'SUZLON', 'TCS', 'TITAN']

    base_path = r'C:\PYTHON\historical_data\CRYPTO_year'

    folders = [
        os.path.join(base_path, symbol, 'output_1day_parquet')
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
                zero_rows.append({
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    **zero_dict
                })

                parquet_path = f"full_optimization_results_{asset_name}_HBullD_{suffix}.parquet"
                df_zero = pd.DataFrame([zero_rows[-1]])
                df_zero.to_parquet(parquet_path, index=False, compression='snappy')

                print(f"  → Wrote empty result for {asset_name}")

            # IMPORTANT: skip the rest of this JSON completely
            continue

        # Generate valid combos once
        print("Generating valid parameter combinations...")
        valid_combos = []
        for combo in product(*ranges):
            hb_hl_upper, hb_hl_lower, hb_ll_upper, hb_ll_lower, \
            hb_gen_slope_upper, hb_gen_slope_lower, hb_gen_macd_slope_upper, \
            hb_gen_macd_slope_lower, date_gap_upper, date_gap_lower = combo

            if (hb_hl_lower >= hb_hl_upper or
                hb_ll_lower >= hb_ll_upper or
                hb_gen_slope_lower >= hb_gen_slope_upper or
                hb_gen_macd_slope_lower >= hb_gen_macd_slope_upper or
                date_gap_lower >= date_gap_upper):
                continue
            valid_combos.append(combo)

        print(f"Generated {len(valid_combos):,} valid combinations")

        for folder_idx, folder_path in enumerate(folders):
            asset_name = symbols[folder_idx]
            print(f"  → Processing {asset_name:12}  ({folder_idx+1:2d}/{len(symbols)})")

            # ────────────────────────────────────────────────
            # EARLY EXIT IF NO COMBINATIONS LEFT
            # ────────────────────────────────────────────────
            if not valid_combos:
                print("    No combinations left → writing zero result")

                zero_dict = {k: 0 for k in param_keys}
                zero_result = {
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    'params': zero_dict
                }

                df_zero = pd.json_normalize([zero_result])
                df_zero = df_zero.rename(columns={f"params.{k}": k for k in param_keys})

                parquet_path = f"full_optimization_results_{asset_name}_HBullD_{suffix}.parquet"
                df_zero.to_parquet(parquet_path, index=False, compression='snappy')

                continue  # ⬅️ skip parquet loading entirely

            # ────────────────────────────────────────────────
            # ONLY REACHED IF valid_combos STILL EXISTS
            # ────────────────────────────────────────────────
            parquet_files = sorted(glob.glob(os.path.join(folder_path, '*.parquet')))
            if not parquet_files:
                print(f"    No parquet files found → skipping")
                continue

            print(f"    Found {len(parquet_files)} parquet files")
            data = parallel_load_files(parquet_files)
            if not data:
                print("    No valid data loaded → skipping")
                continue

            print(f"    Loaded {len(data)} bars")

            # ────────────────────────────────────────────────
            # PROCESS IN COMBO BATCHES TO SAVE RAM
            # ────────────────────────────────────────────────
            all_good_results = []  # Reset for each asset
            batch_num = 1
            param_iter = iter(valid_combos)  # Use pre-filtered valid combos
            while True:
                print(f"    Generating combo batch {batch_num}...")
                batch_combos = list(islice(param_iter, COMBO_BATCH_SIZE))
                if not batch_combos:
                    break

                param_sets_batch = [dict(zip(param_keys, combo)) for combo in batch_combos]
                print(f"    Batch size: {len(param_sets_batch):,}")

                results_batch = run_backtest_for_folder_gpu_parallel(folder_path, param_sets_batch, data)
                good_batch = [r for r in results_batch if r['CAGR'] > 0]
                all_good_results.extend(good_batch)

                print(f"    Processed batch {batch_num} ({len(results_batch):,} results, {len(good_batch):,} positive)")
                batch_num += 1

            # Save and show best after all batches for this asset
            parquet_path = f"full_optimization_results_{asset_name}_HBullD_{suffix}.parquet"

            if all_good_results:
                df_positive = pd.json_normalize(all_good_results)
                df_positive = df_positive.rename(columns={f"params.{k}": k for k in param_keys})
                df_positive.to_parquet(parquet_path, index=False, compression='snappy')

                df_sorted = df_positive.sort_values('CAGR', ascending=False).reset_index(drop=True)
                df_sorted['rank'] = df_sorted.index + 1

                top_n = 1000
                top_df = df_sorted.head(top_n)

                summary_cols = ['rank', 'CAGR', 'Final_Capital', 'Average_ROI', 'Max_Drawdown'] + param_keys
                summary_df = top_df[summary_cols]

                summary_path = f"best_{top_n}_results_{asset_name}_HBullD_{suffix}.csv"
                # summary_df.to_csv(summary_path, index=False)

                # Best result display
                best = df_sorted.iloc[0]
                print("\n" + "=" * 70)
                print(f"               BEST RESULT FOR {asset_name} (suffix: _{suffix})")
                print("=" * 70)
                print(f"Best Final Capital:   ${best['Final_Capital']:,.2f}")
                print(f"Best CAGR:            {best['CAGR']:.2f}%")
                print(f"Average yearly ROI:   {best['Average_ROI']:.2f}%")
                print(f"Max Drawdown:         {best['Max_Drawdown']:.2f}%")
                print("\nOptimal parameters:")
                for k in param_keys:
                    print(f"  {k:25} : {best[k]}")
                print("="*70)

                param_tuple = lambda r: tuple(r['params'][k] for k in param_keys)
                good_tuples = set(param_tuple(r) for r in all_good_results)

            else:
                print("\nNo positive CAGR results for this asset.")
                zero_dict = {k: 0 for k in param_keys}
                zero_result = {
                    'Final_Capital': 0.0,
                    'Average_ROI': 0.0,
                    'CAGR': 0.0,
                    'Max_Drawdown': 0.0,
                    'params': zero_dict
                }
                df_positive = pd.json_normalize([zero_result])
                df_positive = df_positive.rename(columns={f"params.{k}": k for k in param_keys})
                df_positive.to_parquet(parquet_path, index=False, compression='snappy')
                good_tuples = set()

            # Filter valid_combos for next symbol
            valid_combos = [c for c in valid_combos if tuple(c) in good_tuples]
            print(f"    Surviving combinations for next symbol: {len(valid_combos):,}")

    # Now combine per symbol across suffixes
    parameter_cols = [
        'hb_hl_upper', 'hb_hl_lower', 'hb_ll_upper', 'hb_ll_lower',
        'hb_gen_slope_upper', 'hb_gen_slope_lower', 'hb_gen_macd_slope_upper',
        'hb_gen_macd_slope_lower', 'date_gap_upper', 'date_gap_lower'
    ]

    print("\n" + "="*80)
    print("Combining results across all suffixes per symbol...")
    print("="*80)

    for asset_name in symbols:
        all_dfs = []
        for suffix in suffixes:
            file_path = f'full_optimization_results_{asset_name}_HBullD_{suffix}.parquet'
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    all_dfs.append(df)
                    print(f"  Included: {file_path}")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
            else:
                print(f"  Not found: {file_path}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            # Drop duplicates based on parameters, keep first
            combined_df = combined_df.drop_duplicates(subset=parameter_cols, keep='first')
            combined_path = f'full_optimization_results_{asset_name}_HBullD_combined.parquet'
            combined_df.to_parquet(combined_path, index=False, compression='snappy')
            print(f"  → Saved: {combined_path} ({len(combined_df):,} rows)")
        else:
            print(f"  No results found for {asset_name} → skipping combine")

    # ────────────────────────────────────────────────
    # FINAL AUSWERTUNG ON COMBINED DATA
    # ────────────────────────────────────────────────
    print("\nRunning final cross-symbol evaluation on combined results...")
    run_auswertung(symbols, "combined")

    execution_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total execution time: {execution_time:.2f} seconds  ({execution_time/60:.1f} minutes)")
    print(f"{'='*80}")