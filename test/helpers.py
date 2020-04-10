class ReturnThenRun:
    """
    Callable used as a mock side_effect. Returns a value on the first N
    invocations and then calls run_function on subsequent invocations.
    """

    def __init__(self, return_value=None, then_run_function=lambda: None):
        self._invoked = 0
        self.n_times = 1
        self.return_value = return_value
        self.then_run_function = then_run_function

    def __call__(self, *args, **kwargs):
        if self._invoked == self.n_times:
            self.then_run_function()
        else:
            self._invoked += 1
        return self.return_value
