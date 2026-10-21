"""Immutable audit aliases: execute an already approved formula without changing it."""
def build(parameters):
    import factors
    from fea.config import load
    from fea.release import production_specs, definition
    cfg=load()
    original=next(s for s in production_specs(cfg) if s.name==parameters['original_name'])
    if definition(original,cfg)!=parameters['definition_sha256']:
        raise ValueError('Original approved factor changed after audit preregistration')
    # Returning the original callable retains its actual source identity and dependencies.
    return original.fn
