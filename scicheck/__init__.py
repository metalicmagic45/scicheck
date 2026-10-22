"""SciCheck: verifying deep-learning training viability within a scientific workflow.

Module map (paper section -> module):
  §V-A  snapshot representation        -> scicheck.snapshots
  §V-B  judge model (Anomaly Transformer) -> scicheck.anomaly_transformer
  Alg.1 training the judge model        -> scicheck.judge.train_judge
  Alg.2 / §V-D validating a submission  -> scicheck.judge.validate
  §VI-A attack + architectures + data   -> scicheck.attacks, scicheck.models, scicheck.data
  §VI-A accuracy sweep                  -> scicheck.experiments.accuracy
  §VI-B runtime / memory sweeps         -> scicheck.experiments.performance
  Tables IV-VI, Fig. 2                  -> scicheck.report
"""

__version__ = "0.1.0"
