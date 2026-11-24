from .models import DrugInteraction

def check_drug_interactions(drug_names):
    """
    Check interactions between a list of drugs.
    Returns a list of interaction dictionaries.
    """
    if len(drug_names) < 2:
        return []
    
    interactions = []
    drug_names_lower = [d.lower().strip() for d in drug_names if d]
    
    for i in range(len(drug_names_lower)):
        for j in range(i + 1, len(drug_names_lower)):
            drug1 = drug_names_lower[i]
            drug2 = drug_names_lower[j]
            
            # Try direct lookup
            try:
                interaction = DrugInteraction.objects.get(
                    drug_one=drug1, 
                    drug_two=drug2
                )
                interactions.append({
                    'drug1': drug_names[i],
                    'drug2': drug_names[j],
                    'description': interaction.description,
                })
            except DrugInteraction.DoesNotExist:
                # Try reverse order
                try:
                    interaction = DrugInteraction.objects.get(
                        drug_one=drug2,
                        drug_two=drug1
                    )
                    interactions.append({
                        'drug1': drug_names[i],
                        'drug2': drug_names[j],
                        'description': interaction.description,
                    })
                except DrugInteraction.DoesNotExist:
                    pass
    
    return interactions