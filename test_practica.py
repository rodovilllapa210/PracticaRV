"""
Test rápido de PracticaRV.py con solo 1 ISIN
"""

import pandas as pd
import time
from utilities import (
    load_qte_file, load_sts_file,
    filter_magic_numbers, apply_addressability_filter, clean_timestamps_qte,
    create_consolidated_tape, detect_arbitrage_opportunities, detect_rising_edges,
    run_latency_simulation
)

SESSION = '2025-11-07'

# Test with only Santander
TEST_ISIN = 'ES0113900J37'
TEST_NAME = 'Banco Santander'

VENUES = {
    'BME': {'folder': 'BME_2025-11-07', 'mic': 'XMAD'},
    'CBOE': {'folder': 'CBOE_2025-11-07', 'mic': 'CEUX'},
    'AQUIS': {'folder': 'AQUIS_2025-11-07', 'mic': 'AQEU'},
    'TURQUOISE': {'folder': 'TURQUOISE_2025-11-07', 'mic': 'TQEX'}
}

LATENCY_LEVELS = [0, 100, 500, 1000, 2000, 5000, 10000]

print("=" * 80)
print(f"TESTING PracticaRV.py with {TEST_NAME}")
print("=" * 80)

start_time = time.time()

# Step 1: Load and clean
print("\nStep 1: Loading and cleaning data...")
qte_clean = {}

for venue_name, venue_info in VENUES.items():
    df_qte = load_qte_file(TEST_ISIN, venue_name, venue_info, SESSION)
    
    if df_qte is None:
        print(f"  ✗ {venue_name}: No data")
        continue
    
    print(f"  {venue_name}: {len(df_qte):,} quotes", end="")
    
    df_sts = load_sts_file(TEST_ISIN, venue_name, venue_info, SESSION)
    df_qte = filter_magic_numbers(df_qte)
    
    if df_sts is not None:
        df_qte = apply_addressability_filter(df_qte, df_sts, venue_info['mic'])
    
    df_qte = clean_timestamps_qte(df_qte)
    qte_clean[venue_name] = df_qte
    
    print(f" → {len(df_qte):,} clean")

# Step 2 & 3: Consolidated tape and arbitrage
print("\nStep 2 & 3: Consolidated tape and arbitrage detection...")
tape = create_consolidated_tape(qte_clean, TEST_ISIN)
print(f"  Consolidated tape: {len(tape):,} timestamps")

tape = detect_arbitrage_opportunities(tape, list(qte_clean.keys()))
tape = detect_rising_edges(tape)

arb_count = tape['is_arbitrage'].sum()
rising_count = tape['is_rising_edge'].sum()
profit_0 = tape.loc[tape['is_rising_edge'], 'profit'].sum()

print(f"  Arbitrage opportunities: {arb_count:,}")
print(f"  Unique trades: {rising_count:,}")
print(f"  Profit (0 latency): €{profit_0:.2f}")

# Step 4: Latency simulation
print("\nStep 4: Latency simulation...")
results = run_latency_simulation(tape, LATENCY_LEVELS)

print("\nLatency Results:")
for latency_us in LATENCY_LEVELS:
    profit = results[latency_us]
    decay = (profit_0 - profit) / profit_0 * 100 if profit_0 > 0 else 0
    print(f"  {latency_us:6d} μs: €{profit:8.2f} (decay: {decay:5.1f}%)")

elapsed = time.time() - start_time
print(f"\n✓ Test completed in {elapsed:.2f} seconds")
