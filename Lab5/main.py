from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# AF = as frunza
# RF = rege frunza
# RI = rege inima
# RGF = regina frunza
# RGI = regina inima
game_model = BayesianNetwork(
    [
        ("CP1", "CP2"),
        ("CP1", "DP1"),
        ("CP2", "DP2"),
        ("DP1", "DP2"),
    ]
)

# Defining the parameters using CPT

cpd_CP1 = TabularCPD(
    variable="CP1", variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
)
cpd_CP2 = TabularCPD(
    variable="CP2",
    variable_card=5,
    values=[[0, 0.25, 0.25, 0.25, 0.25], [0.25, 0, 0.25, 0.25, 0.25], [0.25, 0.25, 0, 0.25, 0.25], [0.25, 0.25, 0.25, 0, 0.25], [0.25, 0.25, 0.25, 0.25, 0]],
    evidence=["CP1"],
    evidence_card=[5],
)
cpd_DP1 = TabularCPD(
    variable="DP1",
    variable_card=2,
    values=[[0.95, 0.75, 0.5, 0.25, 0.05], [0.05, 0.25, 0.5, 0.75, 0.95]],
    evidence=["CP1"],
    evidence_card=[5],
)
cpd_DP2 = TabularCPD(
    variable="DP2",
    variable_card=2,
    values=[[], []],
    evidence=["CP2", "DP1"],
    evidence_card=[2, 5],
)


# Associating the parameters with the model structure
game_model.add_cpds(
    cpd_CP1, cpd_CP2, cpd_DP1, cpd_DP2
)