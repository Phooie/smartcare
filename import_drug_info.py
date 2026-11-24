import csv
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartCare.settings')
import django
django.setup()

from SmartCare.models import DrugInformation, Medication

def import_drug_info(csv_path):
    """
    Import drug information from CSV file.
    
    CSV columns expected:
    Medicine Name,Composition,Uses,Side_effects,Image URL,Manufacturer,
    Excellent Review %,Average Review %,Poor Review %
    """
    count = 0
    skipped = 0
    matched = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            name = row.get('Medicine Name', '').strip()
            composition = row.get('Composition', '').strip()
            uses = row.get('Uses', '').strip()
            side_effects = row.get('Side_effects', '').strip()
            image_url = row.get('Image URL', '').strip()
            manufacturer = row.get('Manufacturer', '').strip()
            
            try:
                excellent = int(row.get('Excellent Review %', '0').strip() or '0')
            except ValueError:
                excellent = 0
            try:
                average = int(row.get('Average Review %', '0').strip() or '0')
            except ValueError:
                average = 0
            try:
                poor = int(row.get('Poor Review %', '0').strip() or '0')
            except ValueError:
                poor = 0
            
            if not name:
                skipped += 1
                continue
            
            # Try to match with existing medication
            medication = None
            # Try exact match first
            try:
                medication = Medication.objects.get(name__iexact=name)
                matched += 1
            except Medication.DoesNotExist:
                # Try partial match
                meds = Medication.objects.filter(name__icontains=name)
                if meds.exists():
                    medication = meds.first()
                    matched += 1
                else:
                    # Try matching by composition keywords
                    if composition:
                        for comp_part in composition.split('+'):
                            comp_part = comp_part.strip()
                            if len(comp_part) > 3:
                                meds = Medication.objects.filter(name__icontains=comp_part)
                                if meds.exists():
                                    medication = meds.first()
                                    matched += 1
                                    break
            
            # Create or update
            obj, created = DrugInformation.objects.update_or_create(
                name=name,
                defaults={
                    'composition': composition,
                    'uses': uses,
                    'side_effects': side_effects,
                    'image_url': image_url,
                    'manufacturer': manufacturer,
                    'excellent_review_percent': excellent,
                    'average_review_percent': average,
                    'poor_review_percent': poor,
                    'medication': medication,
                }
            )
            
            count += 1
            if count % 100 == 0:
                print(f"Imported {count} drug records... (matched: {matched})")
    
    print(f"\n✅ Completed! Imported {count} drug records. Skipped {skipped} rows. Matched {matched} to existing medications.")

def export_drug_info_to_csv():
    """Export existing drug info to CSV for backup"""
    import csv
    from django.http import HttpResponse
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="drug_info_export.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Medicine Name', 'Composition', 'Uses', 'Side_effects', 
        'Image URL', 'Manufacturer', 'Excellent Review %', 
        'Average Review %', 'Poor Review %'
    ])
    
    for drug in DrugInformation.objects.all():
        writer.writerow([
            drug.name,
            drug.composition,
            drug.uses,
            drug.side_effects,
            drug.image_url,
            drug.manufacturer,
            drug.excellent_review_percent,
            drug.average_review_percent,
            drug.poor_review_percent,
        ])
    
    return response

if __name__ == "__main__":
    # Update the path to your CSV file
    import_drug_info("Medicine_Details.csv")