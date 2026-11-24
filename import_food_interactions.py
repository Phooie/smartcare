import json
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartCare.settings')
import django
django.setup()

from SmartCare.models import FoodInteraction, Medication

def import_food_interactions(json_path):
    """
    Import food interactions from JSON file.
    
    JSON format expected:
    [
        {
            "name": "Drug name",
            "reference": "Reference text",
            "food_interactions": [
                "Interaction text 1",
                "Interaction text 2"
            ]
        }
    ]
    """
    count = 0
    skipped = 0
    matched = 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        drug_name = item.get('name', '').strip()
        reference = item.get('reference', '').strip()
        interactions = item.get('food_interactions', [])
        
        if not drug_name or not interactions:
            skipped += 1
            continue
        
        # Try to find matching medication
        medication = None
        try:
            medication = Medication.objects.get(name__icontains=drug_name)
            matched += 1
        except Medication.DoesNotExist:
            # Try partial match
            meds = Medication.objects.filter(name__icontains=drug_name)
            if meds.exists():
                medication = meds.first()
                matched += 1
        except Medication.MultipleObjectsReturned:
            medication = Medication.objects.filter(name__icontains=drug_name).first()
            matched += 1
        
        for interaction_text in interactions:
            interaction_text = interaction_text.strip()
            if not interaction_text:
                continue
            
            # Determine interaction type based on text
            interaction_type = 'avoid'
            text_lower = interaction_text.lower()
            
            if 'take with food' in text_lower or 'take with a meal' in text_lower:
                interaction_type = 'take_with'
            elif 'take on an empty stomach' in text_lower or 'empty stomach' in text_lower:
                interaction_type = 'take_without'
            elif 'avoid' in text_lower:
                interaction_type = 'avoid'
            elif 'caution' in text_lower:
                interaction_type = 'caution'
            elif 'separate' in text_lower or 'apart' in text_lower:
                interaction_type = 'take_separate'
            elif 'supplement' in text_lower or 'administer' in text_lower:
                interaction_type = 'supplement'
            elif 'drink plenty' in text_lower or 'fluids' in text_lower:
                interaction_type = 'drink_fluids'
            elif 'limit' in text_lower:
                interaction_type = 'limit'
            elif 'increase' in text_lower:
                interaction_type = 'increase'
            elif 'no food' in text_lower or 'no interactions' in text_lower:
                interaction_type = 'no_interaction'
            
            # Determine severity based on text
            severity = 'unknown'
            if any(word in text_lower for word in ['severe', 'serious', 'dangerous', 'fatal']):
                severity = 'severe'
            elif any(word in text_lower for word in ['moderate', 'may']):
                severity = 'moderate'
            elif any(word in text_lower for word in ['mild', 'slight', 'minor']):
                severity = 'mild'
            
            # Create or update
            obj, created = FoodInteraction.objects.update_or_create(
                drug_name=drug_name,
                interaction_text=interaction_text,
                defaults={
                    'medication': medication,
                    'interaction_type': interaction_type,
                    'severity': severity,
                    'reference': reference,
                }
            )
            count += 1
    
    print(f"\n✅ Completed! Imported {count} food interactions from {len(data)} drugs.")
    print(f"   Skipped: {skipped} entries. Matched {matched} to existing medications.")

def export_food_interactions_to_json():
    """Export food interactions to JSON"""
    import json
    from django.http import JsonResponse
    
    data = []
    for interaction in FoodInteraction.objects.select_related('medication').all():
        data.append({
            'name': interaction.drug_name,
            'medication': interaction.medication.name if interaction.medication else None,
            'interaction_type': interaction.get_interaction_type_display(),
            'interaction_text': interaction.interaction_text,
            'severity': interaction.get_severity_display(),
            'reference': interaction.reference,
        })
    
    return JsonResponse(data, safe=False, indent=2)

if __name__ == "__main__":
    # Update the path to your JSON file
    import_food_interactions("Drug to Food interactions Dataset.json")