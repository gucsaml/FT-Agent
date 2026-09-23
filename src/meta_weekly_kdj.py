#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
META 週 K 線 KDJ 計算器

功能：
  1. 透過富途 OpenAPI（futu-api）讀取指定美股近 N 年「日 K 線」（前復權）
  2. 將日 K 線聚合成「週 K 線」（開＝首日開、高＝區間最高、低＝區間最低、
     收＝末日收、量＝區間加總）
  3. 以標準 KDJ(9,3,3) 演算法計算「週線」K、D、J 值
  4. 於終端列印最近若干週結果，並可另存 CSV 與 PNG 圖檔

前置條件：
  - OpenD 已啟動（預設連線 127.0.0.1:11111）
  - 已安裝依賴：pip install futu-api pandas  （matplotlib 為選用，繪圖才需要）

用法範例：
  python3 meta_weekly_kdj.py                          # 預設 US.META 近 2 年
  python3 meta_weekly_kdj.py --code US.META --years 2 --n 9
  python3 meta_weekly_kdj.py --code US.NVDA --years 3 --out ./outputs
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

from futu import OpenQuoteContext, KLType, AuType, RET_OK


def fetch_daily_kline(code, start, end, host="127.0.0.1", port=11111):
    """抓取指定區間日 K 線（前復權），自動翻頁拉完整資料。

    回傳欄位：time_key, open, high, low, close, volume
    """
    ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, data, page_req_key = ctx.request_history_kline(
            code,
            start=start,
            end=end,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            max_count=1000,
        )
        if ret != RET_OK:
            raise RuntimeError(f"取得日 K 線失敗：{data}")

        frames = [data]
        while page_req_key is not None:
            ret, data, page_req_key = ctx.request_history_kline(
                code,
                start=start,
                end=end,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
                max_count=1000,
                page_req_key=page_req_key,
            )
            if ret != RET_OK:
                raise RuntimeError(f"取得日 K 線（翻頁）失敗：{data}")
            if data is not None and not data.empty:
                frames.append(data)

        df = pd.concat(frames, ignore_index=True)
        df = df[["time_key", "open", "high", "low", "close", "volume"]].copy()
        df["time_key"] = pd.to_datetime(df["time_key"])
        df = (
            df.sort_values("time_key")
            .drop_duplicates("time_key")
            .reset_index(drop=True)
        )
        return df
    finally:
        ctx.close()


def daily_to_weekly(daily):
    """日 K → 週 K。美股以週五為週結算；最後一段不足一週者仍獨立成週。"""
    df = daily.set_index("time_key")
    weekly = (
        df.resample("W-FRI")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    return weekly


def calc_kdj(weekly, n=9):
    """計算週線 KDJ（標準 9,3,3）。

    公式：
      RSV = (C - LLV(L,n)) / (HHV(H,n) - LLV(L,n)) * 100
      K   = (2/3) * K_prev + (1/3) * RSV    （初值 50）
      D   = (2/3) * D_prev + (1/3) * K      （初值 50）
      J   = 3K - 2D
    """
    df = weekly.copy()
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()

    span = (high_n - low_n).replace(0, float("nan"))
    rsv = ((df["close"] - low_n) / span * 100).fillna(50.0)

    k = 50.0
    d = 50.0
    k_vals = []
    d_vals = []
    for r in rsv:
        k = (2.0 / 3.0) * k + (1.0 / 3.0) * r
        d = (2.0 / 3.0) * d + (1.0 / 3.0) * k
        k_vals.append(k)
        d_vals.append(d)

    df["K"] = k_vals
    df["D"] = d_vals
    df["J"] = 3.0 * df["K"] - 2.0 * df["D"]
    return df


def plot_result(weekly, code, out_path):
    """繪製收盤價 + KDJ 子圖，存成 PNG。matplotlib 不存在時略過。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[提示] 未安裝 matplotlib，略過繪圖。", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(weekly["time_key"], weekly["close"], label="收盤價", color="#1f77b4")
    ax1.set_title(f"{code} 週 K 線收盤價")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(weekly["time_key"], weekly["K"], label="K", color="#d62728")
    ax2.plot(weekly["time_key"], weekly["D"], label="D", color="#2ca02c")
    ax2.plot(weekly["time_key"], weekly["J"], label="J", color="#9467bd")
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(20, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("週線 KDJ")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"已儲存圖檔：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="讀取美股近 N 年日 K 線，計算週線 KDJ")
    parser.add_argument("--code", default="US.META", help="股票代碼（預設 US.META）")
    parser.add_argument("--years", type=float, default=2.0, help="回溯年數（預設 2）")
    parser.add_argument("--n", type=int, default=9, help="KDJ 的 RSV 週期（預設 9）")
    parser.add_argument("--host", default="127.0.0.1", help="OpenD 主機")
    parser.add_argument("--port", type=int, default=11111, help="OpenD 埠")
    parser.add_argument("--tail", type=int, default=12, help="列印最近幾週（預設 12）")
    parser.add_argument("--out", default=".", help="CSV/PNG 輸出目錄（預設目前目錄）")
    args = parser.parse_args()

    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365.25))
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"讀取 {args.code} 日 K 線：{start_str} ~ {end_str}（前復權）…")
    daily = fetch_daily_kline(
        args.code, start_str, end_str, host=args.host, port=args.port
    )
    if daily.empty:
        print("未取得任何日 K 線資料。")
        sys.exit(1)
    print(f"日 K 線筆數：{len(daily)}")

    weekly = daily_to_weekly(daily)
    weekly = calc_kdj(weekly, n=args.n)
    print(f"週 K 線筆數：{len(weekly)}")

    os.makedirs(args.out, exist_ok=True)
    stem = args.code.replace(".", "_").lower()
    csv_path = os.path.join(args.out, f"{stem}_weekly_kdj.csv")
    png_path = os.path.join(args.out, f"{stem}_weekly_kdj.png")

    out_cols = ["time_key", "open", "high", "low", "close", "volume", "K", "D", "J"]
    weekly[out_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"已儲存 CSV：{csv_path}")

    plot_result(weekly, args.code, png_path)

    print("\n最近幾週 KDJ（收盤價 / K / D / J）：")
    show = weekly[["time_key", "close", "K", "D", "J"]].tail(args.tail).copy()
    show["time_key"] = show["time_key"].dt.strftime("%Y-%m-%d")
    show["close"] = show["close"].round(2)
    show["K"] = show["K"].round(2)
    show["D"] = show["D"].round(2)
    show["J"] = show["J"].round(2)
    print(show.to_string(index=False))

    latest = weekly.iloc[-1]
    print(
        "\n最新一週 KDJ："
        f"K={latest['K']:.2f}, D={latest['D']:.2f}, J={latest['J']:.2f}"
        f"（{latest['time_key']:%Y-%m-%d} 收盤 {latest['close']:.2f}）"
    )






if __name__ == "__main__":
    main()
