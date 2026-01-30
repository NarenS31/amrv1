"""
AMRFinder TSV → R/S prediction wrapper
Used for fair benchmark against ML models
"""

def predict_amrfinder_rs(df, drug):
    """
    df: pandas DataFrame from an AMRFinder TSV
    drug: normalized drug string used in your ML tasks
    returns: "R" or "S"
    """
    from src.amrfinder_drug_rules import AMRFinderPredictor

    predictor = AMRFinderPredictor()
    pred = predictor.predict_genome(df, drug)

    # normalize output
    if isinstance(pred, str):
        return "R" if pred.upper().startswith("R") else "S"
    return "R" if bool(pred) else "S"
