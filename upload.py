import shutil
import os

def process_upload(filename):
    source = f"uploads/{filename}"
    destination = f"data/{filename}"
    
    if os.path.exists(source):
        shutil.move(source, destination)
        print(f"✅ Successfully moved {filename} from uploads to data folder.")
    else:
        print(f"❌ Error: File {filename} not found in uploads folder.")

# Example usage:
# process_upload("purchase_orders.csv")