import os
import re
import csv
import json
import glob
import argparse

def extract_alpaca_eval_metrics(text):
    """Extract AlpacaEval metrics from the log text."""
    # Look for AlpacaEval results in the format of a table with metrics
    alpaca_pattern = r'([^\s]+(?:\/[^\s]+)*)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(?:\d+\.\d+)\s+(?:\d+)\s+(?:\d+)'
    
    matches = re.findall(alpaca_pattern, text)
    
    if not matches:
        return {}
    
    alpaca_results = {}
    for match in matches:
        model_path = match[0].strip()
        lcwr = float(match[1].strip())
        wr = float(match[2].strip())
        
        # Extract model name from path
        if '/' in model_path:
            parts = model_path.split('/')
            for i in range(len(parts)-1, -1, -1):
                if parts[i] and parts[i] != "FINAL":
                    model_name = parts[i]
                    break
            else:
                model_name = model_path
        else:
            model_name = model_path
        
        alpaca_results[model_name] = {
            'alpacaeval_lcwr': lcwr,
            'alpacaeval_wr': wr
        }
    
    return alpaca_results

def extract_log_values(text):
    """Extract only top-level metric values from the log text."""
    results = {}
    
    # Model name extraction
    model_pattern = r"pretrained=([^,]+)"
    model_match = re.search(model_pattern, text)
    if not model_match:  # Skip if no model name found
        return None
    
    model_path = model_match.group(1)
    # Get the model name from the path
    parts = model_path.split("/")
    for i in range(len(parts)-1, -1, -1):
        if parts[i] and parts[i] != "FINAL":
            results["model_name"] = parts[i]
            break
    else:
        results["model_name"] = model_path
    
    # Pattern to match table rows with metrics
    # Match task, version, filter, n-shot, metric, direction, value, stderr
    row_pattern = r"\|([^|]+?)\s*\|\s*(\d+)?\s*\|\s*([^|]+?)\s*\|\s*(\d*)\s*\|\s*([^|]+?)\s*\|\s*([↑↓]?)\s*\|\s*(\d+\.\d+)\s*\|\s*±\s*\|\s*([^|]*)\s*\|"
    
    # Also get the group section metrics (but not subgroups)
    group_pattern = r"\|([^|]+?)\s*\|\s*(\d+)?\s*\|\s*([^|]+?)\s*\|\s*(\d*)\s*\|\s*([^|]+?)\s*\|\s*([↑↓]?)\s*\|\s*(\d+\.\d+)\s*\|\s*±\s*\|\s*(\d+\.\d+)\s*\|"
    
    # Combine both individual metrics and group metrics
    matches = re.findall(row_pattern, text)
    
    for match in matches:
        task = match[0].strip()
        version = match[1].strip() if match[1] else ""
        filter_type = match[2].strip()
        n_shot = match[3].strip() if match[3] else "0"
        metric = match[4].strip()
        direction = match[5].strip()
        value = match[6].strip()
        stderr = match[7].strip()
        
        # Skip header rows, separator rows, or empty tasks
        if not task or task == "Tasks" or task == "Groups" or "--" in task:
            continue
            
        # Skip detailed breakdowns (subtasks)
        if task.startswith("- ") or task.startswith(" "):
            continue
            
        # Create a unique key for this metric
        key = f"{task}-{metric}"
        
        # Store the value
        results[key] = float(value)
        
        # Store stderr if available and not N/A
        if stderr and stderr != "N/A":
            results[f"{key}_stderr"] = float(stderr)
    
    # Extract AlpacaEval metrics and add them to the results if available
    alpaca_metrics = extract_alpaca_eval_metrics(text)
    if results["model_name"] in alpaca_metrics:
        results.update(alpaca_metrics[results["model_name"]])
    
    # Calculate an average of all metrics (excluding AlpacaEval metrics)
    metrics = [v for k, v in results.items() 
              if k != 'model_name' and not k.endswith("_stderr") 
              and not k.startswith("alpacaeval_")]
    
    if metrics:
        results['avg'] = round(sum(metrics) / len(metrics), 4)
    
    return results

def process_log_file(file_path):
    """Process a single log file and extract top-level model metrics."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        results = extract_log_values(log_content)
        if results:
            print(f"Found metrics for model: {results['model_name']} in {file_path}")
            return results
        else:
            print(f"No metrics found in: {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_directory(directory_path):
    """Process all log files in a directory."""
    # List all .log and .txt files in the directory
    log_files = glob.glob(os.path.join(directory_path, "*.log"))
    
    if not log_files:
        print(f"No log files found in {directory_path}")
        return []
    
    print(f"Found {len(log_files)} log files in {directory_path}")
    
    all_results = []
    for log_file in log_files:
        result = process_log_file(log_file)
        if result:
            all_results.append(result)
    
    return all_results

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract top-level metrics from log files.')
    parser.add_argument('path', help='Path to a log file or directory containing log files')
    parser.add_argument('--output', '-o', default='model_metrics', help='Base name for output files (without extension)')
    args = parser.parse_args()
    
    # Check if the path is a directory or a file
    if os.path.isdir(args.path):
        all_results = process_directory(args.path)
    elif os.path.isfile(args.path):
        result = process_log_file(args.path)
        all_results = [result] if result else []
    else:
        print(f"Path not found: {args.path}")
        exit(1)
    
    if not all_results:
        print("No results found.")
        exit(1)
    
    # Get all unique metrics across all results
    all_metrics = set()
    for result in all_results:
        all_metrics.update(key for key in result.keys() 
                          if key != 'model_name' and key != 'avg' and not key.endswith('_stderr'))
    
    # Define field names with model_name first, then AlpacaEval metrics, then other metrics alphabetically, then avg
    alpaca_metrics = [m for m in all_metrics if m.startswith('alpacaeval_')]
    other_metrics = sorted([m for m in all_metrics if not m.startswith('alpacaeval_')])
    fieldnames = ['model_name'] + sorted(alpaca_metrics) + other_metrics + ['avg']
    
    # Write to CSV
    csv_file = f"{args.output}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            # Fill in missing values with None
            row = {field: result.get(field, None) for field in fieldnames}
            writer.writerow(row)
    
    # Also save as JSON for easier programmatic access
    json_file = f"{args.output}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Extracted top-level metrics for {len(all_results)} models")
    print(f"Found {len(all_metrics)} unique metrics")
    print(f"Results saved to {csv_file} and {json_file}")

    # Print a summary table of the models and their average scores
    print("\nSummary of models and average scores:")
    print("-" * 70)
    header = f"{'Model Name':<30} | {'Average Score':<10}"
    if any('alpacaeval_lcwr' in result for result in all_results):
        header += f" | {'LCWR':<10} | {'WR':<10}"
    print(header)
    print("-" * 70)
    
    for result in sorted(all_results, key=lambda x: x.get('avg', 0), reverse=True):
        model_name = result.get('model_name', 'Unknown')
        avg = result.get('avg', 'N/A')
        row = f"{model_name:<30} | {avg if avg == 'N/A' else f'{avg:<10.4f}'}"
        
        if 'alpacaeval_lcwr' in result or 'alpacaeval_wr' in result:
            lcwr = result.get('alpacaeval_lcwr', 'N/A')
            wr = result.get('alpacaeval_wr', 'N/A')
            row += f" | {lcwr if lcwr == 'N/A' else f'{lcwr:<10.2f}'}"
            row += f" | {wr if wr == 'N/A' else f'{wr:<10.2f}'}"
        
        print(row)
    
    print("-" * 70)