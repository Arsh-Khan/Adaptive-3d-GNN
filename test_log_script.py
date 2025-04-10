import re
import pandas as pd
from pathlib import Path

def parse_output_file(file_content):
    """Parse the output file to extract meta epochs and corresponding metrics."""

    pattern = r"finish step (\d+).*?\nacc train: ([\d\.]+), auc train: ([\d\.]+)\nacc val: ([\d\.]+), auc val: ([\d\.]+)\nacc test: ([\d\.]+), auc test: ([\d\.]+)"
    
    results = []
    matches = re.findall(pattern, file_content, re.DOTALL)
    
    for match in matches:
        meta_epoch = int(match[0])
        auc_train = float(match[2])
        auc_val = float(match[4])
        auc_test = float(match[6])
        
        results.append({
            'meta_epoch_index': meta_epoch,
            'auc_train': auc_train,
            'auc_val': auc_val,
            'auc_test': auc_test
        })
    
    return results

def process_files(directory="./test_results/", output_folder="./test_results_processed/"):
    """Process all .out files in the directory and create the output files."""

    out_files = list(Path(directory).glob("*.out"))

    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    all_best_results = []
        
    for file_path in out_files:
        filename = file_path.name.split("_", 2)[-1]
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        results = parse_output_file(content)
            
        if not results:
            print(f"No results found in {filename}, skipping...")
            continue
                
        results_sorted = sorted(results, key=lambda x: x['auc_val'], reverse=True)
            
        df = pd.DataFrame(results_sorted)
        sorted_output_file = f"{output_folder_path}/{filename}_sorted_results.csv"
        df.to_csv(sorted_output_file, index=False)
            
        best_result = results_sorted[0]
        all_best_results.append({
                'input_file': filename,
                'meta_epoch_index': best_result['meta_epoch_index'],
                'best_auc_val': best_result['auc_val'],
                'auc_test': best_result['auc_test']
            })
    else:
        print(f"Processed files successfully.")

    best_df = pd.DataFrame(all_best_results)
    best_output_file = f"{output_folder_path}/best_auc_val_results.csv"
    best_df.to_csv(best_output_file, index=False)
        
    print(f"Processed {len(out_files)} files")
    print(f"Created best results file: {best_output_file}")

if __name__ == "__main__":
    print("Processing output files to extract AUC metrics...")
    process_files(directory='./final_saving_all_output')
    print("Done!")

