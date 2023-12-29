from attrdict import AttrDict
import yaml
import importlib


def dict2cls(d):
    assert {"module", "name", "params"} == set(d.keys())
    module = importlib.import_module(d.module)
    cls = getattr(module, d.name)
    return cls(**d.params) if d.params else cls()


if __name__ == '__main__':
    with open("cfg.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = AttrDict(cfg_dict)

    cfg['scheduler']['params'] = {"initial_learning_rate": 1e-4,
                                 "decay_steps": 100,
                                 "alpha": 0.01
                                 }
    dict2cls(cfg.scheduler)

    optimizer = dict2cls(cfg.optimizer)
    # print(cfg.loss.type)
    # module = importlib.import_module(cfg.optimizer.module)
    # cls = getattr(module, cfg.optimizer.name)
    # loss = cls(**cfg.optimizer.params)
    print("asdf")


