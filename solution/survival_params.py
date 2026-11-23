"""Eleven frozen cohorts fit only through 2025-01-12 (rates per hour).

Alpha is closed-window LOAD incidence; mu excludes the assigned-rotation LOAD.
Numeric values are copied exactly from the Phase 3C fit-only audit.
"""

SURVIVAL_PARAMETERS = {'missed_load': [0.0, 0.0],
 'expired_no_rot': [0.0, 0.0012960803460670934],
 'unscheduled_active': [0.0, 0.0007350460228820691],
 'sched_overdue': [0.23579545454545456, 0.0007299892550461782],
 'sched_init_active': [0.3744955609362389, 0.00044370331006166676],
 'sched_disch_active|24-72h': [0.4117181314330958, 0.0009247161178854214],
 'sched_disch_active|<24h': [0.41233373639661425, 0.0012983760606503725],
 'sched_disch_active|>72h': [0.4119601328903654, 0.000861253385295512],
 'sched_truck_active|24-72h': [0.447887323943662, 0.0004439584782547995],
 'sched_truck_active|<24h': [0.6116504854368932, 0.0006955404842236932],
 'sched_truck_active|>72h': [0.4804804804804805, 0.0010093573508448376]}
