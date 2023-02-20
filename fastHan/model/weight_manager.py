def convert_cws_macro(all_tasks):
    result = dict()
    for task in all_tasks:
        if task.startswith('CWS'):
            result[task] = 'CWS'
        else:
            result[task] = task
    return result


class WeightManagerBase(object):
    def __init__(self, all_tasks, key_mapper=None):
        self.all_tasks = all_tasks
        if key_mapper is not None:
            self.key_mapper = key_mapper
        else:
            self.key_mapper = dict([(x, x) for x in all_tasks])

    def get(self, task):
        task = self.key_mapper.get(task)
        return self.weight.get(task)

    def update(self, task, loss):
        pass


class FixedWeightManager(WeightManagerBase):
    def __init__(self, all_tasks, weight, key_mapper=None):
        super().__init__(all_tasks=all_tasks, key_mapper=key_mapper)
        self.weight = weight


class QueueWeightManager(WeightManagerBase):
    def __init__(self,
                 all_tasks,
                 max_steps=1000,
                 norm_number=1000,
                 key_mapper=None):
        assert norm_number <= max_steps

        super().__init__(all_tasks=all_tasks, key_mapper=key_mapper)
        self.max_steps = max_steps
        self.norm_number = norm_number

        self.weight = dict([(x, 1) for x in self.key_mapper.values()])
        self.task_loss = dict([(x, []) for x in self.key_mapper.values()])

    def update(self, task, loss):
        task = self.key_mapper.get(task)
        current_length = len(self.task_loss.get(task))
        if current_length >= self.max_steps:
            return
        self.task_loss.get(task).append(loss)
        if current_length == self.max_steps - 1:
            self.weight[task] = sum(self.task_loss.get(task)) / self.max_steps


class MomentumWeightManager(WeightManagerBase):
    def __init__(self, all_tasks, beta=0.9, key_mapper=None):
        super().__init__(all_tasks=all_tasks, key_mapper=key_mapper)

        self.beta = beta
        self.weight = dict([(x, 0) for x in self.key_mapper.values()])
        # 直接维护 beta**step
        self.step_norm = dict([(x, 1) for x in self.key_mapper.values()])

    def update(self, task, loss):
        task = self.key_mapper.get(task)
        self.step_norm[task] = self.step_norm[task] * self.beta
        self.weight[task] = self.weight[task] * self.beta + (1 -
                                                             self.beta) * loss

    def get(self, task):
        task = self.key_mapper.get(task)
        return self.weight.get(task) / (1 - self.step_norm[task])


class EnsembledWeightManagers(object):
    def __init__(self, managers):
        for manager in managers:
            assert isinstance(manager[0], WeightManagerBase)
        self.managers = managers

    def get(self, task):
        weight = 1
        for manager, pow in self.managers:
            # print(manager,manager.weight,manager.key_mapper.get(task))
            weight = weight * (manager.get(task)**pow)
        return weight

    def update(self, task, loss):
        for manager, pow in self.managers:
            # print(manager,manager.weight,manager.key_mapper.get(task))
            manager.update(task, loss)
