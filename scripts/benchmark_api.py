#!/usr/bin/env python3
"""Measure end-to-end latency for the API /predict endpoint."""

import os
import sys
import time
import statistics
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Run: pip install requests")
    sys.exit(1)

# ========================= CONFIG =========================
DEFAULT_IMAGE = os.getenv("BENCHMARK_IMAGE")
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
DEFAULT_NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "75"))
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "512"))
DEFAULT_CONF = float(os.getenv("CONF", "0.25"))
DEFAULT_SAVE_ANNOTATED = os.getenv("SAVE_ANNOTATED", "false").lower() == "true"
PROJECT_DIR = Path(__file__).resolve().parent.parent

# ========================= UTILS =========================
def percentile(data: List[float], p: float) -> float:
    """Calculate a percentile from a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]

def format_time(ms: float) -> str:
    """Format a duration in milliseconds."""
    if ms < 1:
        return f"{ms*1000:.2f}us"
    elif ms < 1000:
        return f"{ms:.2f}ms"
    else:
        return f"{ms/1000:.2f}s"

def log(msg: str):
    print(f"[bench_e2e] {msg}")

# ========================= MAIN =========================
def benchmark_e2e(
    image_path: str,
    api_url: str,
    num_requests: int,
    imgsz: int,
    conf: float,
    save_annotated: bool,
    warmup: int = 3
) -> dict:
    """Measure end-to-end latency by sending repeated requests."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    log(f"Image     : {img_path}")
    log(f"API URL   : {api_url}")
    log(f"Requests  : {num_requests}")
    log(f"imgsz     : {imgsz}")
    log(f"conf      : {conf}")
    log(f"save_annot: {save_annotated}")
    log(f"Warmup    : {warmup}")
    
    # Read the image once to avoid including disk I/O in each request.
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    log(f"Image size: {len(image_data) / 1024:.2f} KB")
    
    # Warmup requests are excluded from the results.
    log(f"Warming up ({warmup} requests)...")
    for i in range(warmup):
        try:
            files = {'file': (img_path.name, image_data, 'image/jpeg')}
            data = {
                'imgsz': str(imgsz),
                'conf': str(conf),
                'save_annotated': 'true' if save_annotated else 'false'
            }
            resp = requests.post(api_url, files=files, data=data, timeout=60)
            if resp.status_code == 200:
                log(f"  Warmup {i+1}/{warmup}: OK")
            else:
                log(f"  Warmup {i+1}/{warmup}: FAILED ({resp.status_code})")
        except Exception as e:
            log(f"  Warmup {i+1}/{warmup}: ERROR - {e}")
    
    # Benchmark requests
    log(f"\nStarting benchmark ({num_requests} requests)...")
    latencies: List[float] = []
    errors: List[str] = []
    start_total = time.perf_counter()
    
    for i in range(num_requests):
        try:
            # Prepare the multipart request.
            files = {'file': (img_path.name, image_data, 'image/jpeg')}
            data = {
                'imgsz': str(imgsz),
                'conf': str(conf),
                'save_annotated': 'true' if save_annotated else 'false'
            }
            
            # Measure complete request/response latency.
            t0 = time.perf_counter()
            resp = requests.post(api_url, files=files, data=data, timeout=60)
            t1 = time.perf_counter()
            
            latency_ms = (t1 - t0) * 1000.0
            
            if resp.status_code == 200:
                latencies.append(latency_ms)
                # Report progress at approximately ten-percent intervals.
                if (i + 1) % max(1, num_requests // 10) == 0 or (i + 1) == num_requests:
                    log(f"  [{i+1}/{num_requests}] {format_time(latency_ms)}")
            else:
                error_msg = f"HTTP {resp.status_code}"
                errors.append(error_msg)
                log(f"  [{i+1}/{num_requests}] ERROR: {error_msg}")
                
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            log(f"  [{i+1}/{num_requests}] ERROR: {error_msg}")
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Calculate aggregate metrics.
    if not latencies:
        log("ERROR: No requests completed successfully.")
        return {
            'num_requests': num_requests,
            'success_count': 0,
            'error_count': len(errors),
            'success_rate': 0.0,
            'errors': errors
        }
    
    latencies_sorted = sorted(latencies)
    
    results = {
        'num_requests': num_requests,
        'success_count': len(latencies),
        'error_count': len(errors),
        'success_rate': len(latencies) / num_requests * 100.0,
        'latencies_ms': latencies,
        'mean_ms': statistics.mean(latencies),
        'median_ms': statistics.median(latencies),  # p50
        'p50_ms': percentile(latencies, 50),
        'p90_ms': percentile(latencies, 90),
        'p95_ms': percentile(latencies, 95),
        'p99_ms': percentile(latencies, 99),
        'min_ms': min(latencies),
        'max_ms': max(latencies),
        'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'total_time_s': total_time,
        'req_per_sec': len(latencies) / total_time if total_time > 0 else 0.0,
        'errors': errors[:10] if errors else []
    }
    
    return results

def print_results(results: dict):
    """Print benchmark results to the console."""
    print("\n" + "="*60)
    print("END-TO-END LATENCY BENCHMARK")
    print("="*60)
    
    print("\nSummary:")
    print(f"  Total requests     : {results['num_requests']}")
    print(f"  Successful         : {results['success_count']}")
    print(f"  Errors             : {results['error_count']}")
    print(f"  Success rate       : {results['success_rate']:.2f}%")
    
    if results['success_count'] == 0:
        print("\nNo requests completed successfully.")
        if results.get('errors'):
            print("\nExample errors:")
            for err in results['errors'][:5]:
                print(f"  - {err}")
        return
    
    print("\nLatency metrics:")
    print(f"  Mean               : {format_time(results['mean_ms'])}")
    print(f"  Median (p50)       : {format_time(results['median_ms'])}")
    print(f"  p90                : {format_time(results['p90_ms'])}")
    print(f"  p95                : {format_time(results['p95_ms'])}")
    print(f"  p99                : {format_time(results['p99_ms'])}")
    print(f"  Min                : {format_time(results['min_ms'])}")
    print(f"  Max                : {format_time(results['max_ms'])}")
    print(f"  Std Dev            : {format_time(results['std_ms'])}")
    
    print("\nThroughput:")
    print(f"  Total time         : {results['total_time_s']:.2f}s")
    print(f"  Requests/second    : {results['req_per_sec']:.2f}")
    
    # Display a compact text histogram.
    if len(results['latencies_ms']) > 0:
        latencies = results['latencies_ms']
        print("\nDistribution:")
        bins = [
            (0, 100, "<100ms"),
            (100, 200, "100-200ms"),
            (200, 500, "200-500ms"),
            (500, 1000, "500ms-1s"),
            (1000, float('inf'), ">1s")
        ]
        for min_val, max_val, label in bins:
            count = sum(1 for L in latencies if min_val <= L < max_val)
            pct = count / len(latencies) * 100
            bar = "#" * int(pct / 2)
            print(f"  {label:12} : {count:4d} ({pct:5.1f}%) {bar}")

def save_results_to_csv(results: dict, api_url: str, args, csv_path: Optional[Path] = None):
    """Save benchmark results to CSV files."""
    if pd is None:
        log("pandas is not installed; CSV output is disabled. Run: pip install pandas")
        return
    
    if csv_path is None:
        csv_path = PROJECT_DIR / "runs" / "benchmarks" / "benchmark_api.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a single summary row.
    summary_row = {
        'timestamp': datetime.now().isoformat(),
        'api_url': api_url,
        'image_path': str(args.image),
        'num_requests': results['num_requests'],
        'success_count': results['success_count'],
        'error_count': results['error_count'],
        'success_rate_pct': round(results['success_rate'], 2),
        'mean_ms': round(results['mean_ms'], 3),
        'median_ms': round(results['median_ms'], 3),
        'p50_ms': round(results['p50_ms'], 3),
        'p90_ms': round(results['p90_ms'], 3),
        'p95_ms': round(results['p95_ms'], 3),
        'p99_ms': round(results['p99_ms'], 3),
        'min_ms': round(results['min_ms'], 3),
        'max_ms': round(results['max_ms'], 3),
        'std_ms': round(results['std_ms'], 3),
        'total_time_s': round(results['total_time_s'], 2),
        'req_per_sec': round(results['req_per_sec'], 2),
        'imgsz': args.imgsz,
        'conf': args.conf,
        'save_annotated': args.save_annotated,
        'warmup': args.warmup
    }
    
    df_summary = pd.DataFrame([summary_row])
    
    # Append to the summary file, creating it when needed.
    file_exists = csv_path.exists()
    df_summary.to_csv(csv_path, index=False, mode='a', header=not file_exists)
    
    log(f"Saved summary to: {csv_path}")
    
    # Store per-request latency in a separate file.
    if results.get('latencies_ms') and len(results['latencies_ms']) > 0:
        detail_path = csv_path.with_name(csv_path.stem + "_detail.csv")
        df_detail = pd.DataFrame({
            'timestamp': datetime.now().isoformat(),
            'api_url': api_url,
            'request_id': range(1, len(results['latencies_ms']) + 1),
            'latency_ms': results['latencies_ms']
        })
        
        detail_exists = detail_path.exists()
        df_detail.to_csv(detail_path, index=False, mode='a', header=not detail_exists)
        log(f"Saved {len(results['latencies_ms'])} request records to: {detail_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end latency for the API /predict endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with the default request count
  python scripts/benchmark_api.py --image path/to/image.jpg

  # Set an image and request count
  python scripts/benchmark_api.py --image path/to/image.jpg --num-requests 100

  # Target an API running on another host
  python scripts/benchmark_api.py --image image.jpg --api-url http://localhost:8000/predict

  # Customize image size and confidence
  python scripts/benchmark_api.py --image image.jpg --imgsz 640 --conf 0.25

  # Disable annotated image responses
  python scripts/benchmark_api.py --image image.jpg --no-save-annotated
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        default=DEFAULT_IMAGE,
        required=DEFAULT_IMAGE is None,
        help='Image path (or set BENCHMARK_IMAGE)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default=DEFAULT_API_URL,
        help=f'API endpoint URL (default: {DEFAULT_API_URL})'
    )
    parser.add_argument(
        '--num-requests',
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help=f'Number of requests (default: {DEFAULT_NUM_REQUESTS})'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=DEFAULT_IMGSZ,
        help=f'YOLO image size (default: {DEFAULT_IMGSZ})'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=DEFAULT_CONF,
        help=f'Confidence threshold (default: {DEFAULT_CONF})'
    )
    parser.add_argument(
        '--save-annotated',
        action='store_true',
        default=DEFAULT_SAVE_ANNOTATED,
        help='Include annotated images in responses'
    )
    parser.add_argument(
        '--no-save-annotated',
        dest='save_annotated',
        action='store_false',
        help='Do not include annotated images in responses'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup requests (default: 3)'
    )
    
    args = parser.parse_args()
    
    try:
        results = benchmark_e2e(
            image_path=args.image,
            api_url=args.api_url,
            num_requests=args.num_requests,
            imgsz=args.imgsz,
            conf=args.conf,
            save_annotated=args.save_annotated,
            warmup=args.warmup
        )
        
        print_results(results)
        
        # Save results to CSV.
        save_results_to_csv(results, args.api_url, args)
        
        # Exit code
        sys.exit(0 if results['success_count'] > 0 else 1)
        
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
