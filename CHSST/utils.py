import datetime
class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = datetime.datetime.now()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = datetime.datetime.now() - self.clock[key]
        del self.clock[key]
        return interval.total_seconds()
timer=Timer()

class Arguments:
    dataset_type = None
    dataset_path = None
    lr = 1e-2
    use_cuda = True
    record_path = None
    image_size = 1280
    overlap_threshold = 0.35
    batch_size = 24
    num_workers = 0
    resume = None
    base_net = None
    pretrained_ssd = None
    optimizer_type = 'SGD'
    momentum = 0.9
    weight_decay = 5e-4
    scheduler = 'multi-step'
    milestones = None
    t_max = None
    num_epochs = None
    debug_steps = 100
    validation_epochs = 5
    last_epoch = -1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        res = ''
        for attr in dir(self):
            if attr[:2] == '__':
                continue
            res += '%s:%s\n' % (attr, str(getattr(self, attr)))
        return res