"""Stage 3-C structural defaults; experiment loss weights are CLI inputs."""

temperature = 0.1
optimizer = "AdamW"
detector_trainable = False
paired_preprocessing = "deterministic"
zero_gradient_patience = 10

# Deliberately unspecified: the training command must register these choices.
learning_rate = None
lambda_corr = None
lambda_rec = None
