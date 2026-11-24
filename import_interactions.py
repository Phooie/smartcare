import csv
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartCare.settings')
import django
django.setup()

from SmartCare.models import DrugInteraction

def import_interactions(csv_path):
    count = 0
    skipped = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            drug1 = row.get('Drug 1', '').strip()
            drug2 = row.get('Drug 2', '').strip()
            description = row.get('Interaction Description', '').strip()
            
            if not drug1 or not drug2 or not description:
                skipped += 1
                continue
            
            # Store in both directions for easier lookup
            DrugInteraction.objects.get_or_create(
                drug_one=drug1.lower(),
                drug_two=drug2.lower(),
                defaults={'description': description}
            )
            
            # Also create reverse direction
            DrugInteraction.objects.get_or_create(
                drug_one=drug2.lower(),
                drug_two=drug1.lower(),
                defaults={'description': description}
            )
            
            count += 1
            if count % 1000 == 0:
                print(f"Imported {count} interactions...")
    
    print(f"\n✅ Completed! Imported {count} interactions. Skipped {skipped} rows.")

if __name__ == "__main__":
    import_interactions("db_drug_interactions.csv")  # Change to your file name