from transformers.utils.notebook import NotebookProgressCallback, NotebookTrainingTracker

class _NotebookTrainingTrackerNoTable(NotebookTrainingTracker):
    def write_line(self, values): pass

class NotebookProgressCallbackNoTable(NotebookProgressCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.training_tracker = _NotebookTrainingTrackerNoTable(state.max_steps)