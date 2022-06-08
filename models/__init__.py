import importlib
from .poseDet import PoseDet
from .trainer import Trainer


def build_model(args):
    assert 'backbone' in args
    model_lib = importlib.import_module('models.backbone')
    backbone_args = args.backbone
    model = getattr(model_lib, backbone_args.type)
    backbone_args.pop('type')
    args['backbone'] = model(**backbone_args)

    assert 'head' in args
    model_lib = importlib.import_module('models.head')
    head_args = args.head
    model = getattr(model_lib, head_args.type)
    head_args.pop('type')
    args['head'] = model(**head_args)

    return PoseDet(**args)


def build_perturbations(args):
    lib = importlib.import_module('models.perturbation')
    model = getattr(lib, args.type)
    args.pop('type')
    perturbation = model(**args)

    return perturbation


def build_trainer(args):
    args.model = build_model(args.model)

    return Trainer(**args)
