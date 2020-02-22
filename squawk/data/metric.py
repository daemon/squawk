_METRIC_MAP = {}


class Tracker(object):

    def __init_subclass__(cls, **kwargs):
        cls.id_name = kwargs['name']
        _METRIC_MAP[cls.id_name] = cls

    def update(self, *args):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


class AccuracyTracker(Tracker, name='accuracy'):

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, scores, labels):
        self.num_correct += (scores.max(1)[1] == labels).float().sum().item()
        self.num_total += scores.size(0)

    def reset(self):
        self.num_correct = 0
        self.num_total = 0

    @property
    def value(self):
        return self.num_correct / self.num_total

    @property
    def name(self):
        return 'Accuracy'


class MAPTracker(AccuracyTracker, name='map'):

    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def update(self, scores, labels):
        topk_labels_list = scores.topk(self.precision, 1)[1].tolist()
        self.num_correct += sum(int(lab in topk_labels) for lab, topk_labels in zip(labels.tolist(), topk_labels_list))
        self.num_total += scores.size(0)

    @property
    def name(self):
        return f'MAP@{self.precision}'


def find_metric(name):
    return _METRIC_MAP[name]

