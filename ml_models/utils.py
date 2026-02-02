import pandas as pd
import functools

def with_backend_peek(func):
    """
    Decorator that automatically adds a 'data_peek' and 'schema' 
    to any ML model result based on its specific findings.
    """
    @functools.wraps(func)
    def wrapper(df, *args, **kwargs):
        # 1. Run the original ML model logic
        results = func(df, *args, **kwargs)
        
        # 2. Identification Logic: Find which rows are most relevant
        relevant_indices = []
        
        # Case A: If it's an Anomaly model, peek at the flagged records
        if "anomaly_indices" in results:
            relevant_indices.extend(results["anomaly_indices"])
        elif "flagged_record_indices" in results:
            relevant_indices.extend(results["flagged_record_indices"])
            
        # Case B: If it's a Recommendation model, peek at matched entities
        if "recommendations" in results:
            for rec in results["recommendations"].values():
                relevant_indices.extend(rec.get("matches", []))
                relevant_indices.extend(rec.get("peer_group_matches", []))
        
        # Case C: General Peek (Top 5 outliers based on the target column)
        numeric_cols = df.select_dtypes(include='number').columns
        if not numeric_cols.empty:
            target_col = numeric_cols[-1]
            relevant_indices.extend(df[target_col].nlargest(5).index.tolist())

        # 3. Extract and Clean the Peek
        # Limit to unique indices and max 15 rows to save tokens
        unique_indices = list(dict.fromkeys(relevant_indices))[:15]
        peek_df = df.iloc[unique_indices]

        # 4. Inject the metadata into the results
        results["backend_context"] = {
            "columns_present": list(df.columns),
            "evidence_rows": peek_df.to_dict(orient='records'),
            "data_summary": df.describe().to_dict() # Adds statistical grounding
        }
        
        return results
    return wrapper