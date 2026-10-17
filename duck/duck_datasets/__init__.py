"""Dataset registry. To add a dataset, write a module with a SPEC and list it here."""

from . import coyo700m, coyo_hd_11m, pixelprose

REGISTRY = {spec.name: spec for spec in (coyo700m.SPEC, coyo_hd_11m.SPEC, pixelprose.SPEC)}
