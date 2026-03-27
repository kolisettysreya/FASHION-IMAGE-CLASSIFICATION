import pandas as pd
import os
import shutil

# === Step 1: Load CSV file ===
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'styles.csv'))
df = pd.read_csv(csv_path, on_bad_lines='skip')

df = df.dropna(subset=['id', 'masterCategory'])

# === Step 2: Paths ===
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'images'))
output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'masterCategory'))

# Create output folder if it doesn't exist
os.makedirs(output_base, exist_ok=True)

# === Step 3: Loop through rows and copy ===
count = 0
for _, row in df.iterrows():
    try:
        img_id = int(row['id'])
        category = str(row['masterCategory']).strip()

        src_path = os.path.join(image_dir, f"{img_id}.jpg")
        dst_folder = os.path.join(output_base, category)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, f"{img_id}.jpg")

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            count += 1

    except Exception as e:
        print(f"❌ Error processing ID {row['id']}: {e}")

print(f"✅ Done! {count} images copied into folders by masterCategory.")



