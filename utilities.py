"""
Utilities for Arbitrage Detection Practice
Contains all helper functions for data loading, cleaning, and analysis
"""

import pandas as pd
import numpy as np
import glob
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

MAGIC_NUMBERS = [
    666666.666,  # Unquoted/Unknown
    999999.999,  # Market Order (At Best)
    999999.989,  # At Open Order
    999999.988,  # At Close Order
    999999.979,  # Pegged Order
    999999.123,  # Unquoted/Unknown
    0.0          # Zero price (invalid)
]

VALID_TRADING_STATUS = {
    'AQEU': [5308427],
    'XMAD': [5832713, 5832756],
    'CEUX': [12255233],
    'TQEX': [7608181]
}


# ============================================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================================

def load_qte_file(isin, venue_name, venue_info, session, data_folder='DATA_BIG'):
    """
    Load a QTE (quotes) file for a specific ISIN and venue.
    
    Args:
        isin: ISIN code
        venue_name: Venue name (e.g., 'BME')
        venue_info: Dict with 'folder' and 'mic'
        session: Trading session date
        data_folder: Root data folder
        
    Returns:
        DataFrame with order book snapshots or None if not found
    """
    folder = venue_info['folder']
    mic = venue_info['mic']
    
    pattern = f'{data_folder}/{folder}/QTE_{session}_{isin}_*_{mic}_*.csv.gz'
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    try:
        df = pd.read_csv(files[0], sep=';', compression='gzip')
        
        # Select only needed columns
        cols_needed = ['epoch', 'sequence', 'px_bid_0', 'px_ask_0', 
                      'qty_bid_0', 'qty_ask_0', 'mic']
        df = df[cols_needed].copy()
        
        # Add metadata
        df['venue'] = venue_name
        df['isin'] = isin
        
        return df
    
    except Exception as e:
        print(f"    ✗ Error loading {venue_name}: {e}")
        return None


def load_sts_file(isin, venue_name, venue_info, session, data_folder='DATA_BIG'):
    """
    Load an STS (trading status) file for a specific ISIN and venue.
    
    Args:
        isin: ISIN code
        venue_name: Venue name
        venue_info: Dict with 'folder' and 'mic'
        session: Trading session date
        data_folder: Root data folder
        
    Returns:
        DataFrame with market status updates or None if not found
    """
    folder = venue_info['folder']
    mic = venue_info['mic']
    
    pattern = f'{data_folder}/{folder}/STS_{session}_{isin}_*_{mic}_*.csv.gz'
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    try:
        df = pd.read_csv(files[0], sep=';', compression='gzip')
        
        # Select only needed columns
        cols_needed = ['epoch', 'sequence', 'market_trading_status', 'mic']
        df = df[cols_needed].copy()
        
        # Add metadata
        df['venue'] = venue_name
        df['isin'] = isin
        
        return df
    
    except Exception as e:
        return None


def filter_magic_numbers(df, magic_numbers=MAGIC_NUMBERS):
    """
    Remove rows where bid or ask prices are 'magic numbers' (invalid prices).
    Also applies sanity checks for valid market data.
    
    Args:
        df: DataFrame with px_bid_0, px_ask_0 columns
        magic_numbers: List of magic number values to filter
        
    Returns:
        Filtered DataFrame
    """
    original_len = len(df)
    
    # Filter out magic numbers in prices
    for magic in magic_numbers:
        df = df[df['px_bid_0'] != magic]
        df = df[df['px_ask_0'] != magic]
    
    # Filter out NaN prices
    df = df.dropna(subset=['px_bid_0', 'px_ask_0'])
    
    # Filter out zero or NaN quantities
    df = df[(df['qty_bid_0'] > 0) & (df['qty_ask_0'] > 0)]
    df = df.dropna(subset=['qty_bid_0', 'qty_ask_0'])
    
    # Sanity check: bid should be less than ask
    df = df[df['px_bid_0'] < df['px_ask_0']]
    
    # Additional check: filter unreasonably high prices (likely missed magic numbers)
    # Most stocks trade under 100 EUR
    df = df[(df['px_bid_0'] < 100) & (df['px_ask_0'] < 100)]
    
    return df


def apply_addressability_filter(qte_df, sts_df, mic, valid_status_codes=VALID_TRADING_STATUS):
    """
    Merge QTE data with STS data to determine which quotes are 'addressable'
    (i.e., the market was in continuous trading).
    
    Args:
        qte_df: Quote DataFrame
        sts_df: Status DataFrame
        mic: Market Identifier Code
        valid_status_codes: Dict mapping MIC to valid status codes
        
    Returns:
        Filtered DataFrame with only addressable quotes
    """
    valid_statuses = valid_status_codes.get(mic, [])
    
    if not valid_statuses:
        # No valid status codes defined, keep all quotes
        return qte_df
    
    # Sort both dataframes by epoch
    qte_sorted = qte_df.sort_values('epoch').copy()
    sts_sorted = sts_df.sort_values('epoch').copy()
    
    # Merge as-of: for each quote, find the most recent trading status
    merged = pd.merge_asof(
        qte_sorted,
        sts_sorted[['epoch', 'market_trading_status']],
        on='epoch',
        direction='backward'
    )
    
    # Filter to keep only addressable quotes
    addressable = merged[merged['market_trading_status'].isin(valid_statuses)].copy()
    
    return addressable


def clean_timestamps_qte(df):
    """
    Apply the nanosecond trick to create unique timestamps for quotes
    that occur at the same microsecond.
    
    Args:
        df: DataFrame with epoch and sequence columns
        
    Returns:
        DataFrame with unique timestamp index
    """
    # Sort by epoch and sequence
    df = df.sort_values(by=['epoch', 'sequence'], ascending=[True, True])
    
    # Convert to timestamp
    temp_ts = pd.to_datetime(df['epoch'], unit='us')
    
    # Add nanosecond offset for duplicates
    offset_ns = df.groupby('epoch').cumcount()
    
    # Create unique timestamp
    df['ts'] = temp_ts + pd.to_timedelta(offset_ns, unit='ns')
    
    # Set as index
    df.set_index('ts', inplace=True)
    
    return df


# ============================================================================
# STEP 2: CONSOLIDATED TAPE
# ============================================================================

def create_consolidated_tape(qte_dict, isin):
    """
    Create a consolidated tape for a single ISIN across all venues.
    
    Args:
        qte_dict: Dict mapping venue_name -> DataFrame
        isin: ISIN code
        
    Returns:
        DataFrame with columns for each venue's bid/ask prices and quantities
    """
    # Combine all venue data
    all_data = []
    for venue_name, df in qte_dict.items():
        df_copy = df.copy()
        df_copy['venue_name'] = venue_name
        all_data.append(df_copy)
    
    if not all_data:
        return None
    
    # Concatenate all venues
    combined = pd.concat(all_data)
    combined = combined.sort_index()
    
    # Pivot to create columns for each venue
    tape = pd.DataFrame(index=combined.index.unique())
    
    for venue_name in qte_dict.keys():
        venue_data = combined[combined['venue_name'] == venue_name]
        
        # Add bid/ask prices
        tape[f'bid_{venue_name}'] = venue_data['px_bid_0']
        tape[f'ask_{venue_name}'] = venue_data['px_ask_0']
        
        # Add bid/ask quantities
        tape[f'bid_qty_{venue_name}'] = venue_data['qty_bid_0']
        tape[f'ask_qty_{venue_name}'] = venue_data['qty_ask_0']
    
    # Forward fill to propagate last known prices
    tape = tape.ffill()
    
    # Drop rows where we don't have data from all venues yet
    tape = tape.dropna()
    
    return tape


# ============================================================================
# STEP 3: ARBITRAGE DETECTION
# ============================================================================

def detect_arbitrage_opportunities(tape, venues):
    """
    Detect arbitrage opportunities in the consolidated tape.
    
    Args:
        tape: Consolidated tape DataFrame
        venues: List of venue names
        
    Returns:
        DataFrame with arbitrage detection columns added
    """
    # Get all bid and ask columns
    bid_cols = [f'bid_{v}' for v in venues]
    ask_cols = [f'ask_{v}' for v in venues]
    
    # Find max bid and min ask across venues
    tape['max_bid'] = tape[bid_cols].max(axis=1)
    tape['min_ask'] = tape[ask_cols].min(axis=1)
    
    # Find which venues have the max bid and min ask
    tape['max_bid_venue'] = tape[bid_cols].idxmax(axis=1).str.replace('bid_', '')
    tape['min_ask_venue'] = tape[ask_cols].idxmin(axis=1).str.replace('ask_', '')
    
    # Calculate spread (arbitrage opportunity exists when spread > 0)
    tape['spread'] = tape['max_bid'] - tape['min_ask']
    
    # Get available quantities at the best prices
    tape['bid_qty_at_max'] = tape.apply(
        lambda row: row[f"bid_qty_{row['max_bid_venue']}"], axis=1
    )
    tape['ask_qty_at_min'] = tape.apply(
        lambda row: row[f"ask_qty_{row['min_ask_venue']}"], axis=1
    )
    
    # Available quantity is the minimum of the two
    tape['available_qty'] = tape[['bid_qty_at_max', 'ask_qty_at_min']].min(axis=1)
    
    # Calculate theoretical profit (in EUR)
    tape['profit'] = tape['spread'] * tape['available_qty']
    
    # Flag arbitrage opportunities (spread > 0)
    tape['is_arbitrage'] = tape['spread'] > 0
    
    return tape


def detect_rising_edges(tape):
    """
    Detect rising edges of arbitrage opportunities.
    A rising edge is when an arbitrage opportunity appears after not existing.
    
    Args:
        tape: Consolidated tape with arbitrage detection
        
    Returns:
        DataFrame with is_rising_edge column added
    """
    # Create a boolean mask for arbitrage opportunities
    has_arbitrage = tape['spread'] > 0
    
    # Detect rising edges: True when current is arbitrage but previous wasn't
    tape['is_rising_edge'] = has_arbitrage & ~has_arbitrage.shift(1, fill_value=False)
    
    return tape


def apply_quality_filters(tape, filter_config):
    """
    Apply quality filters to remove suspicious or unrealistic arbitrage opportunities.
    
    Filters applied:
    1. Duration: Remove opportunities too short (< min_duration_ms) or too long (> max_duration_ms)
    2. Spread: Remove opportunities with unrealistic spreads (> max_spread_eur)
    3. Quantity: Remove opportunities with insufficient tradable quantity (< min_tradable_qty)
    4. Depth anomaly: Remove opportunities with depth=100% but very low quantity (suspicious)
    
    Args:
        tape: Consolidated tape with rising edges detected
        filter_config: Dictionary with filter parameters
        
    Returns:
        Tuple: (filtered_tape, filter_stats)
    """
    # Get only rising edges (opportunities)
    opportunities = tape[tape['is_rising_edge']].copy()
    
    if len(opportunities) == 0:
        return tape, {
            'initial_opportunities': 0,
            'filtered_by_min_duration': 0,
            'filtered_by_max_duration': 0,
            'filtered_by_spread': 0,
            'filtered_by_quantity': 0,
            'filtered_by_depth_anomaly': 0,
            'final_opportunities': 0,
            'total_filtered': 0,
            'filter_rate_pct': 0.0
        }
    
    initial_count = len(opportunities)
    
    # Calculate duration of each opportunity (time until next rising edge or end)
    opportunities['next_edge_time'] = opportunities.index.to_series().shift(-1)
    opportunities['duration_ms'] = (
        (opportunities['next_edge_time'] - opportunities.index).dt.total_seconds() * 1000
    )
    
    # For the last opportunity, use a default duration
    opportunities['duration_ms'] = opportunities['duration_ms'].fillna(filter_config['min_duration_ms'] * 2)
    
    # Track filtering statistics
    filter_stats = {
        'initial_opportunities': initial_count,
        'filtered_by_min_duration': 0,
        'filtered_by_max_duration': 0,
        'filtered_by_spread': 0,
        'filtered_by_quantity': 0,
        'filtered_by_depth_anomaly': 0,
        'final_opportunities': 0
    }
    
    # Create mask for valid opportunities
    valid_mask = pd.Series(True, index=opportunities.index)
    
    # Filter 1: Minimum duration
    min_duration_mask = opportunities['duration_ms'] >= filter_config['min_duration_ms']
    filter_stats['filtered_by_min_duration'] = (~min_duration_mask).sum()
    valid_mask &= min_duration_mask
    
    # Filter 2: Maximum duration (stale quotes)
    max_duration_mask = opportunities['duration_ms'] <= filter_config['max_duration_ms']
    filter_stats['filtered_by_max_duration'] = (~max_duration_mask).sum()
    valid_mask &= max_duration_mask
    
    # Filter 3: Maximum spread
    spread_mask = opportunities['spread'] <= filter_config['max_spread_eur']
    filter_stats['filtered_by_spread'] = (~spread_mask).sum()
    valid_mask &= spread_mask
    
    # Filter 4: Minimum tradable quantity
    qty_mask = opportunities['available_qty'] >= filter_config['min_tradable_qty']
    filter_stats['filtered_by_quantity'] = (~qty_mask).sum()
    valid_mask &= qty_mask
    
    # Filter 5: Depth anomaly (depth=100% but very low quantity)
    # This catches cases where spread=100% of bid-ask but quantity is suspiciously low
    depth_pct = (opportunities['spread'] / opportunities['spread'].replace(0, 1)) * 100
    depth_anomaly_mask = ~((depth_pct > 99) & (opportunities['available_qty'] <= filter_config['depth_100_qty_max']))
    filter_stats['filtered_by_depth_anomaly'] = (~depth_anomaly_mask).sum()
    valid_mask &= depth_anomaly_mask
    
    # Apply filters to the original tape
    filtered_tape = tape.copy()
    
    # Mark filtered opportunities as not rising edges
    invalid_opportunities = opportunities[~valid_mask].index
    filtered_tape.loc[invalid_opportunities, 'is_rising_edge'] = False
    
    filter_stats['final_opportunities'] = valid_mask.sum()
    filter_stats['total_filtered'] = initial_count - filter_stats['final_opportunities']
    filter_stats['filter_rate_pct'] = (filter_stats['total_filtered'] / initial_count * 100) if initial_count > 0 else 0
    
    return filtered_tape, filter_stats


# ============================================================================
# STEP 4: LATENCY SIMULATION
# ============================================================================

def simulate_latency(tape, latency_us):
    """
    Simulate execution latency by looking up actual profit at T + latency.
    OPTIMIZED VERSION using vectorized operations.
    
    Args:
        tape: Consolidated tape with rising edges detected
        latency_us: Latency in microseconds
        
    Returns:
        Total realized profit at this latency level
    """
    # Get all rising edge timestamps
    rising_edges = tape[tape['is_rising_edge']].copy()
    
    if len(rising_edges) == 0:
        return 0.0
    
    # Calculate execution times (vectorized)
    execution_times = rising_edges.index + pd.Timedelta(microseconds=latency_us)
    
    # Create a DataFrame with execution times and signal profits
    signals = pd.DataFrame({
        'execution_time': execution_times,
        'signal_profit': rising_edges['profit'].values
    })
    
    # Prepare tape for efficient lookup
    tape_lookup = tape[['spread', 'available_qty']].copy()
    
    # Use merge_asof for efficient batch lookup
    # This is MUCH faster than iterating with asof()
    execution_states = pd.merge_asof(
        signals.sort_values('execution_time'),
        tape_lookup.reset_index().rename(columns={'ts': 'time'}),
        left_on='execution_time',
        right_on='time',
        direction='backward'
    )
    
    # Calculate actual profit at execution time
    # Only count profit if spread > 0 at execution time
    execution_states['actual_profit'] = 0.0
    mask = execution_states['spread'] > 0
    execution_states.loc[mask, 'actual_profit'] = (
        execution_states.loc[mask, 'spread'] * 
        execution_states.loc[mask, 'available_qty']
    )
    
    # Sum up all realized profits
    total_profit = execution_states['actual_profit'].sum()
    
    return total_profit


def run_latency_simulation(tape, latency_levels):
    """
    Run latency simulation for multiple latency levels.
    
    Args:
        tape: Consolidated tape with arbitrage detection
        latency_levels: List of latency values in microseconds
        
    Returns:
        Dict mapping latency -> realized profit
    """
    results = {}
    
    for latency_us in latency_levels:
        profit = simulate_latency(tape, latency_us)
        results[latency_us] = profit
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_isins(data_folder='DATA_BIG', min_venues=4):
    """
    Scan data folder to find ISINs available in at least min_venues.
    
    Args:
        data_folder: Root data folder
        min_venues: Minimum number of venues required
        
    Returns:
        List of ISINs available in at least min_venues
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        return []
    
    isin_venue_count = {}
    
    for venue_folder in data_path.iterdir():
        if not venue_folder.is_dir():
            continue
            
        qte_files = list(venue_folder.glob('QTE_*.csv.gz'))
        
        for file in qte_files:
            # Parse filename: QTE_<session>_<isin>_<ticker>_<mic>_<part>.csv.gz
            parts = file.stem.replace('.csv', '').split('_')
            if len(parts) >= 3:
                isin = parts[2]
                isin_venue_count[isin] = isin_venue_count.get(isin, 0) + 1
    
    # Filter ISINs available in at least min_venues
    available_isins = [isin for isin, count in isin_venue_count.items() 
                      if count >= min_venues]
    
    return sorted(available_isins)


def format_currency(value):
    """Format value as currency string."""
    return f"€{value:,.2f}"


def format_percentage(value):
    """Format value as percentage string."""
    return f"{value:.2f}%"


# ============================================================================
# VENUE-TO-VENUE ANALYSIS
# ============================================================================

def analyze_venue_pairs(tape, venues):
    """
    Analyze arbitrage opportunities between each pair of venues.
    
    For each pair (Venue A, Venue B), we look for opportunities where:
    - Buy at Venue B (min ask) and Sell at Venue A (max bid)
    - This is directional: A->B is different from B->A
    
    Args:
        tape: Consolidated tape with arbitrage detection
        venues: List of venue names
        
    Returns:
        DataFrame with venue pair analysis
    """
    import itertools
    
    # Get only rising edges (unique opportunities)
    opportunities = tape[tape['is_rising_edge']].copy()
    
    if len(opportunities) == 0:
        return pd.DataFrame()
    
    # Analyze each directional pair
    pair_stats = []
    
    for sell_venue, buy_venue in itertools.permutations(venues, 2):
        # Find opportunities where we sell at sell_venue and buy at buy_venue
        # This means: sell_venue has max_bid and buy_venue has min_ask
        mask = (opportunities['max_bid_venue'] == sell_venue) & \
               (opportunities['min_ask_venue'] == buy_venue)
        
        pair_opps = opportunities[mask]
        
        if len(pair_opps) > 0:
            total_profit = pair_opps['profit'].sum()
            count = len(pair_opps)
            avg_profit = total_profit / count if count > 0 else 0
            avg_spread = pair_opps['spread'].mean()
            avg_qty = pair_opps['available_qty'].mean()
            
            pair_stats.append({
                'sell_venue': sell_venue,
                'buy_venue': buy_venue,
                'direction': f"{sell_venue}→{buy_venue}",
                'opportunities': count,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'avg_spread': avg_spread,
                'avg_quantity': avg_qty
            })
    
    if not pair_stats:
        return pd.DataFrame()
    
    df = pd.DataFrame(pair_stats)
    df = df.sort_values('total_profit', ascending=False)
    
    return df


def create_venue_pair_matrix(pair_analysis, venues, metric='total_profit'):
    """
    Create a matrix showing venue pair statistics.
    
    Args:
        pair_analysis: DataFrame from analyze_venue_pairs
        venues: List of venue names
        metric: Which metric to show ('total_profit', 'opportunities', 'avg_profit')
        
    Returns:
        DataFrame matrix with venues as rows/columns
    """
    if len(pair_analysis) == 0:
        return pd.DataFrame()
    
    # Create empty matrix
    matrix = pd.DataFrame(0.0, index=venues, columns=venues)
    
    # Fill matrix
    for _, row in pair_analysis.iterrows():
        sell_venue = row['sell_venue']
        buy_venue = row['buy_venue']
        value = row[metric]
        matrix.loc[sell_venue, buy_venue] = value
    
    return matrix
