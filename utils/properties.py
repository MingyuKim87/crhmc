from enum import Enum

class Integrator(Enum):
    EXPLICIT       = 1
    IMPLICIT       = 2
    S3             = 3
    SPLITTING      = 4
    SPLITTING_RAND = 5
    SPLITTING_KMID = 6


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3
    IDENTITY = 4


def get_metric_properties(args):
    # Metric
    if args.metric == "HESSIAN":
        metric = Metric.HESSIAN
    elif args.metric == "SOFTABS":
        metric = Metric.SOFTABS
    elif args.metric == "JACOBIAN_DIAG":
        metric = Metric.JACOBIAN_DIAG
    elif args.metric == "IDENTITY":
        metric = Metric.IDENTITY
    else:
        NotImplementedError()
        
    return {"metric" : metric}