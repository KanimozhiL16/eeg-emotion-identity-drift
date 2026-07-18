
# Drift-invariant loss block for future model training
# Use this inside the training step after computing identity embeddings z_id
# and emotion labels e.

def pairwise_cosine_matrix(z):
    z = torch.nn.functional.normalize(z, dim=1)
    return z @ z.T

def drift_invariant_loss(z_id, y_subject, y_emotion, margin=0.10):
    """
    Penalizes same-subject embeddings from different emotions when they drift apart.
    z_id: identity embedding [B, D]
    y_subject: subject label [B]
    y_emotion: emotion/cognitive-state label [B]
    """
    sim = pairwise_cosine_matrix(z_id)
    same_subject = y_subject[:, None].eq(y_subject[None, :])
    diff_emotion = ~y_emotion[:, None].eq(y_emotion[None, :])
    mask = same_subject & diff_emotion
    if mask.sum() == 0:
        return z_id.new_tensor(0.0)
    # Want high cross-emotion similarity for the same subject.
    return torch.relu((1.0 - margin) - sim[mask]).mean()

# Example total loss:
# loss = arcface_loss + 0.5 * supcon_loss + 0.2 * drift_invariant_loss(z_id, subject, emotion)
