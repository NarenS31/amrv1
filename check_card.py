import json
import pandas as pd

# Load CARD database
with open('/Users/narensara11/amr-klebsiella/data/card-data/card.json', 'r') as f:
    card = json.load(f)

print("="*80)
print("CARD DATABASE LOADED")
print("="*80)
print(f"Total entries: {len(card)}")

# Get resistance genes
genes = []
for aro_id, entry in card.items():
    if aro_id.isdigit():
        genes.append({
            'aro': aro_id,
            'name': entry.get('ARO_name', ''),
            'category': entry.get('ARO_category', {}).get('category_aro_name', '')
        })

genes_df = pd.DataFrame(genes)
print(f"\nResistance genes: {len(genes_df)}")
print(genes_df.head(10))
