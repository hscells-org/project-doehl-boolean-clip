from transformers.utils.notebook import NotebookProgressCallback, NotebookTrainingTracker
from transformers.integrations import WandbCallback
import numpy as np
from collections import defaultdict

class _NotebookTrainingTrackerNoTable(NotebookTrainingTracker):
    def write_line(self, values): pass

class NotebookProgressCallbackNoTable(NotebookProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.training_tracker = _NotebookTrainingTrackerNoTable(state.max_steps)

class WandbCallbackAveraged(WandbCallback):
    def __init__(self):
        self.global_step = 0
        self.logs = {}
        super().__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        self.logs.update(logs)
        if self.global_step is not state.global_step:
            self.global_step = state.global_step
            if self.logs is not None:
                groups = defaultdict(list)
                for key, val in self.logs.items():
                    parts = key.split('_', 2)
                    if len(parts) == 3: _, _, metric = parts
                    else: continue
                    groups[metric].append(val)

                # for each suffix with more than one entry, compute and insert the mean
                for metric, vals in groups.items():
                    if len(vals) > 1:
                        self.logs[f"eval_mean_{metric}"] = float(np.mean(vals))
            super().on_log(args, state, control, model, self.logs, **kwargs)
            self.logs = {}