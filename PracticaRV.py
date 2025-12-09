"""
Práctica de Arbitraje en Mercados Financieros
Detección de oportunidades de arbitraje cross-venue con simulación de latencia

Autor: Rodolfo Villena Lapaz
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import utilities
from utilities import (
    load_qte_file, load_sts_file,
    filter_magic_numbers, apply_addressability_filter, clean_timestamps_qte,
    create_consolidated_tape, detect_arbitrage_opportunities, detect_rising_edges,
    apply_quality_filters,
    run_latency_simulation, format_currency, format_percentage,
    analyze_venue_pairs, create_venue_pair_matrix
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SESSION = '2025-11-07'

# Configuration: Use all available ISINs or just a selection?
USE_ALL_ISINS = False  # Set to True to process all ISINs (slower but complete)

# Selected ISINs for analysis (5 major Spanish stocks) - used if USE_ALL_ISINS = False
ISINS_TO_PROCESS_MANUAL = {
    'ES0113900J37': 'Banco Santander',
    'ES0113211835': 'Inditex', 
    'ES0144580Y14': 'Telefónica',
    'ES0178430E18': 'Iberdrola',
    'ES0113679I37': 'BBVA'
}

# Visualization settings
MAX_ISINS_TO_PLOT = 10  # Maximum number of ISINs to show in individual plots

# Define the venues and their corresponding MICs
VENUES = {
    'BME': {'folder': 'BME_2025-11-07', 'mic': 'XMAD'},
    'CBOE': {'folder': 'CBOE_2025-11-07', 'mic': 'CEUX'},
    'AQUIS': {'folder': 'AQUIS_2025-11-07', 'mic': 'AQEU'},
    'TURQUOISE': {'folder': 'TURQUOISE_2025-11-07', 'mic': 'TQEX'}
}

# Latency levels to simulate (in microseconds)
LATENCY_LEVELS = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 
                  10000, 15000, 20000, 30000, 50000, 100000]

DATA_FOLDER = 'DATA_BIG'

# ============================================================================
# QUALITY FILTERS FOR ARBITRAGE OPPORTUNITIES
# ============================================================================
# These filters remove suspicious or unrealistic opportunities

FILTER_CONFIG = {
    "min_duration_ms": 5.0,          # Minimum reasonable duration (5ms)
    "max_duration_ms": 60_000.0,     # Maximum duration - stale quotes (60s)
    "max_spread_eur": 50.0,          # Maximum reasonable spread (€50)
    "min_tradable_qty": 5,           # Minimum acceptable quantity (5 shares)
    "depth_100_qty_max": 3,          # Depth=100% but qty<3 → suspicious
}

ENABLE_QUALITY_FILTERS = True  # Set to False to disable filters


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("PRÁCTICA DE ARBITRAJE EN MERCADOS FINANCIEROS")
    print("=" * 80)
    print(f"Session: {SESSION}")
    
    # Determine which ISINs to process
    if USE_ALL_ISINS:
        print("Mode: Processing ALL available ISINs")
        from utilities import get_available_isins
        available_isins = get_available_isins(DATA_FOLDER, min_venues=4)
        
        if not available_isins:
            print("ERROR: No ISINs found in DATA_BIG folder")
            return None, None, None
        
        # Create dict with generic names
        ISINS_TO_PROCESS = {isin: f"ISIN_{i+1}" for i, isin in enumerate(available_isins)}
        print(f"Found {len(ISINS_TO_PROCESS)} ISINs available in all 4 venues")
    else:
        print("Mode: Processing SELECTED ISINs only")
        ISINS_TO_PROCESS = ISINS_TO_PROCESS_MANUAL
        print(f"ISINs to process: {len(ISINS_TO_PROCESS)}")
    
    print(f"Venues: {list(VENUES.keys())}")
    print(f"Latency levels: {len(LATENCY_LEVELS)}")
    print("=" * 80)
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: DATA LOADING & CLEANING
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 80)
    
    qte_clean = {}
    
    for isin, name in ISINS_TO_PROCESS.items():
        print(f"\n{name} ({isin}):")
        print("-" * 60)
        
        qte_clean[isin] = {}
        
        for venue_name, venue_info in VENUES.items():
            # Load QTE file
            df_qte = load_qte_file(isin, venue_name, venue_info, SESSION, DATA_FOLDER)
            
            if df_qte is None:
                print(f"  ✗ {venue_name}: No QTE data")
                continue
            
            print(f"  {venue_name}: {len(df_qte):,} quotes loaded", end="")
            
            # Load STS file
            df_sts = load_sts_file(isin, venue_name, venue_info, SESSION, DATA_FOLDER)
            
            # Filter magic numbers
            df_qte = filter_magic_numbers(df_qte)
            
            # Apply addressability filter if STS data available
            if df_sts is not None:
                mic = venue_info['mic']
                df_qte = apply_addressability_filter(df_qte, df_sts, mic)
            
            # Clean timestamps
            df_qte = clean_timestamps_qte(df_qte)
            
            qte_clean[isin][venue_name] = df_qte
            
            print(f" → {len(df_qte):,} clean quotes")
    
    # Summary
    total_quotes = sum(len(df) for isin_data in qte_clean.values() 
                      for df in isin_data.values())
    print(f"\n✓ Step 1 Complete: {total_quotes:,} clean quotes loaded")
    
    # ========================================================================
    # STEP 2 & 3: CONSOLIDATED TAPE & ARBITRAGE DETECTION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2 & 3: CONSOLIDATED TAPE & ARBITRAGE DETECTION")
    print("=" * 80)
    
    consolidated_tapes = {}
    arbitrage_results = {}
    
    for isin, name in ISINS_TO_PROCESS.items():
        print(f"\n{name} ({isin}):")
        
        # Create consolidated tape
        tape = create_consolidated_tape(qte_clean[isin], isin)
        
        if tape is None or len(tape) == 0:
            print("  ✗ No consolidated tape created")
            continue
        
        print(f"  Consolidated tape: {len(tape):,} timestamps")
        
        # Detect arbitrage
        tape = detect_arbitrage_opportunities(tape, list(qte_clean[isin].keys()))
        tape = detect_rising_edges(tape)
        
        # Apply quality filters if enabled
        if ENABLE_QUALITY_FILTERS:
            tape, filter_stats = apply_quality_filters(tape, FILTER_CONFIG)
            print(f"  Quality filters: {filter_stats['total_filtered']} removed ({filter_stats['filter_rate_pct']:.1f}%)")
        
        consolidated_tapes[isin] = tape
        
        # Calculate statistics
        arb_count = tape['is_arbitrage'].sum()
        rising_edge_count = tape['is_rising_edge'].sum()
        total_profit_0_latency = tape.loc[tape['is_rising_edge'], 'profit'].sum()
        
        arbitrage_results[isin] = {
            'name': name,
            'total_timestamps': len(tape),
            'arbitrage_opportunities': arb_count,
            'unique_trades': rising_edge_count,
            'profit_0_latency': total_profit_0_latency,
            'avg_profit_per_trade': total_profit_0_latency / rising_edge_count if rising_edge_count > 0 else 0,
            'arbitrage_pct': arb_count / len(tape) * 100 if len(tape) > 0 else 0
        }
        
        print(f"  Arbitrage opportunities: {arb_count:,} ({arb_count/len(tape)*100:.2f}%)")
        print(f"  Unique trades: {rising_edge_count:,}")
        print(f"  Profit (0 latency): {format_currency(total_profit_0_latency)}")
    
    print(f"\n✓ Step 2 & 3 Complete: Arbitrage detection finished")
    
    # ========================================================================
    # STEP 3.5: VENUE-TO-VENUE ANALYSIS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3.5: VENUE-TO-VENUE PAIR ANALYSIS")
    print("=" * 80)
    
    # Aggregate venue pair analysis across all ISINs
    all_pair_stats = []
    
    for isin, name in ISINS_TO_PROCESS.items():
        if isin not in consolidated_tapes:
            continue
        
        tape = consolidated_tapes[isin]
        pair_analysis = analyze_venue_pairs(tape, list(VENUES.keys()))
        
        if len(pair_analysis) > 0:
            pair_analysis['isin'] = isin
            pair_analysis['name'] = name
            all_pair_stats.append(pair_analysis)
    
    # Combine all ISINs
    if all_pair_stats:
        venue_pairs_df = pd.concat(all_pair_stats, ignore_index=True)
        
        # Aggregate by venue pair
        venue_pair_summary = venue_pairs_df.groupby(['sell_venue', 'buy_venue', 'direction']).agg({
            'opportunities': 'sum',
            'total_profit': 'sum',
            'avg_profit': 'mean',
            'avg_spread': 'mean',
            'avg_quantity': 'mean'
        }).reset_index()
        
        venue_pair_summary = venue_pair_summary.sort_values('total_profit', ascending=False)
        
        print("\nTop 10 Most Profitable Venue Pairs:")
        print("-" * 80)
        top_pairs = venue_pair_summary.head(10).copy()
        top_pairs['total_profit_fmt'] = top_pairs['total_profit'].apply(format_currency)
        top_pairs['avg_profit_fmt'] = top_pairs['avg_profit'].apply(format_currency)
        print(top_pairs[['direction', 'opportunities', 'total_profit_fmt', 'avg_profit_fmt']].to_string(index=False))
        
        # Save to CSV
        venue_pair_summary.to_csv('venue_pairs_analysis.csv', index=False)
        print("\n✓ Venue pair analysis saved to 'venue_pairs_analysis.csv'")
    else:
        venue_pair_summary = pd.DataFrame()
        print("⚠ No venue pair data available")
    
    print(f"\n✓ Step 3.5 Complete: Venue pair analysis finished")
    
    # ========================================================================
    # STEP 4: LATENCY SIMULATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: LATENCY SIMULATION")
    print("=" * 80)
    
    latency_results = {}
    
    for isin, name in ISINS_TO_PROCESS.items():
        if isin not in consolidated_tapes:
            continue
        
        print(f"\n{name} ({isin}):")
        print(f"  Simulating {len(LATENCY_LEVELS)} latency levels...", end=" ")
        
        tape = consolidated_tapes[isin]
        results = run_latency_simulation(tape, LATENCY_LEVELS)
        
        latency_results[isin] = results
        
        print("✓")
        
        # Show sample results
        print(f"  0 μs: {format_currency(results[0])}")
        print(f"  1000 μs: {format_currency(results[1000])}")
        print(f"  10000 μs: {format_currency(results[10000])}")
    
    print(f"\n✓ Step 4 Complete: Latency simulation finished")
    
    # ========================================================================
    # RESULTS & VISUALIZATIONS
    # ========================================================================
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # ========================================================================
    # 1. THE "MONEY TABLE"
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("THE MONEY TABLE: Total Realized Profit by Latency")
    print("=" * 80)
    
    # Create money table
    money_table_data = []
    
    for isin, name in ISINS_TO_PROCESS.items():
        if isin not in latency_results:
            continue
        
        row = {'ISIN': isin, 'Name': name}
        
        for latency_us in LATENCY_LEVELS:
            profit = latency_results[isin].get(latency_us, 0)
            row[f'{latency_us}μs'] = profit
        
        money_table_data.append(row)
    
    money_table = pd.DataFrame(money_table_data)
    
    # Add total row
    total_row = {'ISIN': 'TOTAL', 'Name': 'All ISINs'}
    for latency_us in LATENCY_LEVELS:
        total_profit = sum(latency_results[isin].get(latency_us, 0) 
                          for isin in latency_results.keys())
        total_row[f'{latency_us}μs'] = total_profit
    
    money_table = pd.concat([money_table, pd.DataFrame([total_row])], ignore_index=True)
    
    # Format for display
    money_table_display = money_table.copy()
    for col in money_table_display.columns:
        if 'μs' in col:
            money_table_display[col] = money_table_display[col].apply(format_currency)
    
    print(money_table_display.to_string(index=False))
    
    # Save to CSV
    money_table.to_csv('money_table.csv', index=False)
    print("\n✓ Money table saved to 'money_table.csv'")
    
    # ========================================================================
    # 2. TOP 5 MOST PROFITABLE ISINs (at 0 latency)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TOP 5 MOST PROFITABLE ISINs (0 Latency)")
    print("=" * 80)
    
    top_isins = []
    for isin, results in arbitrage_results.items():
        top_isins.append({
            'ISIN': isin,
            'Name': results['name'],
            'Unique Trades': results['unique_trades'],
            'Total Profit': results['profit_0_latency'],
            'Avg Profit/Trade': results['avg_profit_per_trade'],
            'Arb %': results['arbitrage_pct']
        })
    
    top_isins_df = pd.DataFrame(top_isins)
    top_isins_df = top_isins_df.sort_values('Total Profit', ascending=False).head(5)
    
    # Format for display
    top_isins_display = top_isins_df.copy()
    top_isins_display['Total Profit'] = top_isins_display['Total Profit'].apply(format_currency)
    top_isins_display['Avg Profit/Trade'] = top_isins_display['Avg Profit/Trade'].apply(format_currency)
    top_isins_display['Arb %'] = top_isins_display['Arb %'].apply(format_percentage)
    
    print(top_isins_display.to_string(index=False))
    
    # Sanity check
    print("\n" + "-" * 80)
    print("SANITY CHECK:")
    print("-" * 80)
    for _, row in top_isins_df.iterrows():
        profit = row['Total Profit']
        trades = row['Unique Trades']
        avg = row['Avg Profit/Trade']
        
        if profit > 1000:
            print(f"⚠ {row['Name']}: High profit ({format_currency(profit)}) - verify data quality")
        elif trades == 0:
            print(f"⚠ {row['Name']}: No trades detected - check addressability filter")
        elif avg < 0.01:
            print(f"⚠ {row['Name']}: Very low avg profit ({format_currency(avg)}) - might be noise")
        else:
            print(f"✓ {row['Name']}: Profit looks reasonable ({format_currency(profit)}, {trades} trades)")
    
    # ========================================================================
    # 3. THE DECAY CHART
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GENERATING DECAY CHARTS")
    print("=" * 80)
    
    # Calculate total profits for all latency levels
    total_profits = []
    for latency_us in LATENCY_LEVELS:
        total_profit = sum(latency_results[isin].get(latency_us, 0) 
                          for isin in latency_results.keys())
        total_profits.append(total_profit)
    
    profit_0 = total_profits[0]
    profit_1ms = total_profits[LATENCY_LEVELS.index(1000)]
    profit_10ms = total_profits[LATENCY_LEVELS.index(10000)]
    
    decay_1ms = (profit_0 - profit_1ms) / profit_0 * 100 if profit_0 > 0 else 0
    decay_10ms = (profit_0 - profit_10ms) / profit_0 * 100 if profit_0 > 0 else 0
    
    # ========================================================================
    # Chart 1: Total Profit Decay (Main Chart)
    # ========================================================================
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(LATENCY_LEVELS, total_profits, marker='o', linewidth=2.5, 
             markersize=8, color='steelblue', label='Total Profit', zorder=3)
    ax1.fill_between(LATENCY_LEVELS, total_profits, alpha=0.3, color='steelblue')
    
    ax1.set_xlabel('Latency (microseconds)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Profit (€)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Profit Decay with Increasing Latency ({len(ISINS_TO_PROCESS)} ISINs)', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Add annotations for key latency points
    ax1.axvline(x=1000, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='1ms')
    ax1.axvline(x=10000, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='10ms')
    
    # Add text box with decay statistics
    textstr = f'Decay at 1ms: {decay_1ms:.1f}%\nDecay at 10ms: {decay_10ms:.1f}%\n\nTotal ISINs: {len(ISINS_TO_PROCESS)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('decay_chart_total.png', dpi=300, bbox_inches='tight')
    print("✓ Total decay chart saved to 'decay_chart_total.png'")
    plt.close()
    
    # ========================================================================
    # Chart 2: Top ISINs Decay (Only show top profitable ISINs)
    # ========================================================================
    
    # Get top ISINs by profit
    top_isins_for_plot = sorted(
        [(isin, arbitrage_results[isin]['profit_0_latency'], arbitrage_results[isin]['name']) 
         for isin in latency_results.keys()],
        key=lambda x: x[1],
        reverse=True
    )[:MAX_ISINS_TO_PLOT]
    
    if len(top_isins_for_plot) > 0:
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        
        # Use a colormap for better distinction
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_isins_for_plot)))
        
        for idx, (isin, profit_0, name) in enumerate(top_isins_for_plot):
            profits = [latency_results[isin].get(latency_us, 0) for latency_us in LATENCY_LEVELS]
            ax2.plot(LATENCY_LEVELS, profits, marker='o', linewidth=2, 
                    markersize=5, label=f"{name} (€{profit_0:.2f})", 
                    alpha=0.8, color=colors[idx])
        
        ax2.set_xlabel('Latency (microseconds)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Profit (€)', fontsize=13, fontweight='bold')
        ax2.set_title(f'Profit Decay by ISIN (Top {len(top_isins_for_plot)} Most Profitable)', 
                     fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9, ncol=2 if len(top_isins_for_plot) > 5 else 1)
        
        plt.tight_layout()
        plt.savefig('decay_chart_by_isin.png', dpi=300, bbox_inches='tight')
        print(f"✓ Top {len(top_isins_for_plot)} ISINs decay chart saved to 'decay_chart_by_isin.png'")
        plt.close()
    
    # ========================================================================
    # Chart 3: Distribution of Profits at Key Latencies
    # ========================================================================
    
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    key_latencies = [0, 1000, 10000]
    key_labels = ['0 μs (Instant)', '1 ms', '10 ms']
    
    for idx, (latency_us, label) in enumerate(zip(key_latencies, key_labels)):
        ax = axes[idx]
        
        profits_at_latency = [latency_results[isin].get(latency_us, 0) 
                             for isin in latency_results.keys()]
        profits_at_latency = [p for p in profits_at_latency if p > 0]  # Only positive profits
        
        if len(profits_at_latency) > 0:
            ax.hist(profits_at_latency, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Profit (€)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Number of ISINs', fontsize=11, fontweight='bold')
            ax.set_title(f'Profit Distribution at {label}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_profit = np.mean(profits_at_latency)
            median_profit = np.median(profits_at_latency)
            ax.axvline(mean_profit, color='red', linestyle='--', linewidth=2, label=f'Mean: €{mean_profit:.2f}')
            ax.axvline(median_profit, color='green', linestyle='--', linewidth=2, label=f'Median: €{median_profit:.2f}')
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('profit_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Profit distribution chart saved to 'profit_distribution.png'")
    plt.close()
    
    # ========================================================================
    # Chart 4: Venue Pair Analysis
    # ========================================================================
    
    if len(venue_pair_summary) > 0:
        print("\nGenerating venue pair analysis charts...")
        
        # Create matrices for different metrics
        profit_matrix = create_venue_pair_matrix(venue_pair_summary, list(VENUES.keys()), 'total_profit')
        opps_matrix = create_venue_pair_matrix(venue_pair_summary, list(VENUES.keys()), 'opportunities')
        
        # Chart 4a: Heatmap of Total Profit by Venue Pair
        fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Profit heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(profit_matrix.values, cmap='YlOrRd', aspect='auto')
        
        ax1.set_xticks(np.arange(len(profit_matrix.columns)))
        ax1.set_yticks(np.arange(len(profit_matrix.index)))
        ax1.set_xticklabels(profit_matrix.columns, fontsize=11, fontweight='bold')
        ax1.set_yticklabels(profit_matrix.index, fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Buy Venue', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sell Venue', fontsize=12, fontweight='bold')
        ax1.set_title('Total Profit by Venue Pair (€)\nSell at Row, Buy at Column', 
                     fontsize=13, fontweight='bold')
        
        # Add text annotations
        for i in range(len(profit_matrix.index)):
            for j in range(len(profit_matrix.columns)):
                value = profit_matrix.iloc[i, j]
                if value > 0:
                    text = ax1.text(j, i, f'€{value:.0f}',
                                   ha="center", va="center", color="black" if value < profit_matrix.values.max()/2 else "white",
                                   fontsize=9, fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='Total Profit (€)')
        
        # Opportunities heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(opps_matrix.values, cmap='Blues', aspect='auto')
        
        ax2.set_xticks(np.arange(len(opps_matrix.columns)))
        ax2.set_yticks(np.arange(len(opps_matrix.index)))
        ax2.set_xticklabels(opps_matrix.columns, fontsize=11, fontweight='bold')
        ax2.set_yticklabels(opps_matrix.index, fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('Buy Venue', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sell Venue', fontsize=12, fontweight='bold')
        ax2.set_title('Number of Opportunities by Venue Pair\nSell at Row, Buy at Column', 
                     fontsize=13, fontweight='bold')
        
        # Add text annotations
        for i in range(len(opps_matrix.index)):
            for j in range(len(opps_matrix.columns)):
                value = opps_matrix.iloc[i, j]
                if value > 0:
                    text = ax2.text(j, i, f'{int(value)}',
                                   ha="center", va="center", color="black" if value < opps_matrix.values.max()/2 else "white",
                                   fontsize=9, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='Number of Opportunities')
        
        plt.tight_layout()
        plt.savefig('venue_pairs_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Venue pairs heatmap saved to 'venue_pairs_heatmap.png'")
        plt.close()
        
        # Chart 4b: Bar chart of top venue pairs
        fig5, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top pairs by profit
        ax1 = axes[0]
        top_10_profit = venue_pair_summary.head(10)
        colors_profit = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_10_profit)))
        
        bars1 = ax1.barh(range(len(top_10_profit)), top_10_profit['total_profit'], color=colors_profit)
        ax1.set_yticks(range(len(top_10_profit)))
        ax1.set_yticklabels(top_10_profit['direction'], fontsize=10)
        ax1.set_xlabel('Total Profit (€)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Most Profitable Venue Pairs', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, top_10_profit['total_profit'])):
            ax1.text(value, bar.get_y() + bar.get_height()/2, f' €{value:.2f}',
                    va='center', fontsize=9, fontweight='bold')
        
        # Top pairs by opportunities
        ax2 = axes[1]
        top_10_opps = venue_pair_summary.nlargest(10, 'opportunities')
        colors_opps = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_10_opps)))
        
        bars2 = ax2.barh(range(len(top_10_opps)), top_10_opps['opportunities'], color=colors_opps)
        ax2.set_yticks(range(len(top_10_opps)))
        ax2.set_yticklabels(top_10_opps['direction'], fontsize=10)
        ax2.set_xlabel('Number of Opportunities', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Venue Pairs by Number of Opportunities', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, top_10_opps['opportunities'])):
            ax2.text(value, bar.get_y() + bar.get_height()/2, f' {int(value)}',
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('venue_pairs_top10.png', dpi=300, bbox_inches='tight')
        print("✓ Top 10 venue pairs chart saved to 'venue_pairs_top10.png'")
        plt.close()
    
    print("✓ All charts generated successfully")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    total_trades = sum(r['unique_trades'] for r in arbitrage_results.values())
    total_profit_0 = sum(r['profit_0_latency'] for r in arbitrage_results.values())
    
    print(f"Total ISINs processed: {len(ISINS_TO_PROCESS)}")
    print(f"Total venues: {len(VENUES)}")
    print(f"Total unique trading opportunities: {total_trades:,}")
    print(f"Total profit (0 latency): {format_currency(total_profit_0)}")
    print(f"Total profit (1ms latency): {format_currency(total_profits[LATENCY_LEVELS.index(1000)])}")
    print(f"Total profit (10ms latency): {format_currency(total_profits[LATENCY_LEVELS.index(10000)])}")
    print(f"\nProcessing time: {elapsed_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("✓ PRÁCTICA COMPLETADA")
    print("=" * 80)
    
    return money_table, top_isins_df, latency_results


if __name__ == "__main__":
    money_table, top_isins, latency_results = main()
