"""Exact finite discounted-MDP planning with inspectable residual certificates."""
from .core import audit,evaluate_policy,value_iteration,policy_iteration
__all__=['audit','evaluate_policy','value_iteration','policy_iteration']

from .robust import robust_audit,robust_value_iteration
__all__ += ['robust_audit','robust_value_iteration']
