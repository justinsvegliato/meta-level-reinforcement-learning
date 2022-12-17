class MeanTimeDependentComputationalReturnMetric:

    def __init__(self, name: str = 'MeanTimeDependentComputationalReturn'):
        self.name = name
        self.value = None
        self.count = 0

    def __call__(self, value):
        self.count += 1
        if self.value is None:
            self.value = value
        else:
            self.value = (self.value * (self.count - 1) + value) / self.count

    def result(self):
        return self.value

    def reset(self):
        self.value = None
        self.count = 0


class MeanFinalPolicyValueMetric:

    def __init__(self, name: str = 'FinalPolicyValue'):
        self.name = name
        self.value = None
        self.count = 0

    def __call__(self, value):
        self.count += 1
        if self.value is None:
            self.value = value
        else:
            self.value = (self.value * (self.count - 1) + value) / self.count

    def result(self):
        return self.value

    def reset(self):
        self.value = None
        self.count = 0
