#!/usr/bin/env python3
"""Write a function that selects the best pruning value ccp_alpha
for a set of trained decision trees.
    """


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    This function first identifies the model(s) that achieve
    the highest test accuracy.
    """
    # Tier 1: highest test accuracy
    max_test = max(test_scores)
    candidates = [i for i, s in enumerate(test_scores) if s == max_test]

    if len(candidates) > 1:
        # Tier 2: smallest train-test gap (best generalization)
        gaps = [abs(train_scores[i] - test_scores[i]) for i in candidates]
        min_gap = min(gaps)
        candidates = [i for i, g in zip(candidates, gaps) if g == min_gap]

    if len(candidates) > 1:
        # Tier 3: largest ccp_alpha (simplest tree)
        best_idx = max(candidates, key=lambda i: ccp_alphas[i])
    else:
        best_idx = candidates[0]

    return ccp_alphas[best_idx], clfs[best_idx]
