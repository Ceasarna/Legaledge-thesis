import os
import glob
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set this to the root of your logs folder
LOGS_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "logs")
OUTPUT_CSV = os.path.join(LOGS_DIR, "extracted_tb_times.csv")

# Only process folders containing this date
DATE_FILTER = "20260423_"

def extract_durations(base_dir):
    search_pattern = os.path.join(base_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(search_pattern, recursive=True)

    if not event_files:
        print(f"No TensorBoard logs found in {base_dir}")
        return

    print(f"Applying filter: '{DATE_FILTER}'")
    
    # We will store the extracted data here before saving
    csv_data = []

    for event_file in event_files:
        if DATE_FILTER not in event_file:
            continue

        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            scalar_tags = ea.Tags().get('scalars', [])
            all_times = []
            
            for tag in scalar_tags:
                events = ea.Scalars(tag)
                all_times.extend([e.wall_time for e in events])
            
            if all_times:
                first_time = min(all_times)
                last_time = max(all_times)
                duration_seconds = last_time - first_time
                
                folder_name = os.path.dirname(event_file)
                short_name = folder_name.replace(base_dir, "").lstrip("\\/")
                if not short_name:
                    short_name = "root"
                
                m, s = divmod(duration_seconds, 60)
                h, m = divmod(m, 60)
                formatted_time = f"{int(h)}h {int(m)}m {int(s)}s"

                if duration_seconds > 1.0:
                    # Add to our list to save later
                    csv_data.append([short_name, round(duration_seconds, 2), formatted_time])
                    print(f"Processed: {short_name} -> {formatted_time}")

        except Exception as e:
            pass

    # --- NEW: Save to CSV ---
    if csv_data:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(["Log Directory", "Duration (s)", "Formatted Time"])
            # Write all the rows
            writer.writerows(csv_data)
        
        print("\n" + "="*50)
        print(f"✅ Successfully saved {len(csv_data)} runs to:")
        print(OUTPUT_CSV)
        print("="*50)
    else:
        print("No valid runs found to save.")

if __name__ == "__main__":
    print(f"Scanning {LOGS_DIR} for TensorBoard logs...")
    extract_durations(LOGS_DIR)