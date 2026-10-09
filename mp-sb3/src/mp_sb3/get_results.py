from .experiment import Experiment, load_many
exp = Experiment.from_dir("results/core")
exp.summary()      # one row per run: final return, final bias (mean of last 5 evaluations)
exp.load("evals")  # all evaluations, for the learning and bias curves