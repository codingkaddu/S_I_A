import sys
content = open(r'C:\Users\Rudra\Desktop\Mars\predict.py', 'r', encoding='utf-8').read()

lines = content.split('\n')
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith('class SIAPredictor:'):
        start_idx = i
    if start_idx != -1 and line.startswith('# =============================================================================') and 'EVIDENCE DOSSIER BUILDER' in lines[i+1]:
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    for i in range(start_idx, end_idx):
        lines[i] = '# ' + lines[i]

    heuristic_code = '''
class HeuristicPredictor:
    """
    Rule-based fallback inference engine using simple keyword matching.
    """
    def __init__(self, model_dir=None):
        print("[SIA] Initialising HeuristicPredictor (Fallback Mode)")
        self.max_length = 128
        self.train_config = {}

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        # Keywords for heuristic matching
        heuristic_keywords = ['urgent', 'critical', 'fail', 'broken', 'breach', 'outage', 'down', 'severe', 'asap', 'error', 'hacked', 'stolen']
        
        for text in texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in heuristic_keywords):
                pred_label = 1
                prob = 0.85
            else:
                pred_label = 0
                prob = 0.15
            
            results.append({
                "predicted_label": pred_label,
                "confidence": max(prob, 1 - prob),
                "prob_mismatch": prob,
            })
        return results

# Fallback alias for existing code
SIAPredictor = HeuristicPredictor
'''
    lines.insert(end_idx, heuristic_code)

    # Note: We don't need to change type hints because we alias SIAPredictor = HeuristicPredictor
    open(r'C:\Users\Rudra\Desktop\Mars\predict.py', 'w', encoding='utf-8').write('\n'.join(lines))
    print('predict.py updated successfully.')
else:
    print('Failed to find SIAPredictor block in predict.py')
