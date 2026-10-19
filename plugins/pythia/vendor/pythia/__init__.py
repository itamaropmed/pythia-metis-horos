"""Public Pythia API."""

from .api import certify
from .contracts import (
    Certificate, ClaimTier, DependenceGuardSpec, FoldMatrix, GuardSpec, PairMatrix,
    RangeDependenceGuardSpec, ReplicateGuardSpec, Representation, TaskCard,
)
from .certificate_graph import (
    CompiledCertificate,
    GapEvidence,
    LowerEvidence,
    compile_certificate_graph,
)
from .evidence_adapters import (
    EvidenceContext,
    gap_evidence_from_cap,
    gap_evidence_from_finite_cmi,
    gap_evidence_from_finite_decision,
    gap_evidence_from_profile,
    lower_evidence_from_result,
    structural_zero_gap,
)
from .certificate_manifest import (
    canonical_manifest_hash,
    compile_certificate_manifest,
    compile_certificate_manifest_file,
)
from .core.dependence import dependence_profile
from .core.dependence_bounds import (
    bounded_cluster_pairwise_bound,
    bounded_cluster_sensitivity_curve,
    bounded_sample_pairwise_bound,
    catoni_cluster_pairwise_bound,
    empirical_bernstein_patient_pairwise_bound,
    median_of_means_cluster_pairwise_bound,
    sharp_pairwise_population_bound,
)
from .information_order import (
    RiskInterval,
    close_information_order,
    decompose_achieved_gain,
    information_value_interval,
)
from .representation_falsification import (
    RepresentationInterventionAudit,
    audit_representation_intervention,
)
from .core.replicates import (
    paired_replicate_lower_bound,
    paired_replicate_ensemble_sensitivity,
    paired_replicate_sensitivity_curve,
)
from .core.mae_family_envelope import (
    decision_pair_group_envelope,
    geometry_routed_decision_envelope,
    mae_pair_group_envelope,
)
from .core.moment_saa import (
    MomentSAAResult,
    catoni_second_moment_saa_lower_bound,
    cantelli_saa_lower_bound,
    location_saa_score_variance_cap,
)
from .core.truncated_saa import (
    TruncatedSAACertificate,
    dkelly_lower_mean,
    truncated_saa_portfolio,
)
from .core.finite_cell_saa import FiniteCellSAAResult, finite_cell_saa_lower_bound
from .core.finite_cell_lattice import (
    FiniteCellLatticeResult,
    finite_cell_information_lattice,
)
from .core.finite_refinement_test import (
    FiniteRefinementTest,
    finite_refinement_permutation_test,
)
from .core.fixed_policy_value import (
    FixedPolicyInformationValue,
    fixed_policy_information_value_lower_bound,
)
from .core.clustered_cell_saa import (
    ClusteredFiniteCellSAAResult,
    ClusteredFixedPolicyInformationValue,
    clustered_finite_cell_saa_lower_bound,
    clustered_fixed_policy_information_value_lower_bound,
)
from .core.clustered_block_saa import clustered_block_saa_portfolio
from .core.dominated_saa import DominatedSAACertificate, dominated_saa_lower_bound
from .core.finite_cmi import (
    FiniteCMIUpperBound,
    FiniteDecisionGapUpperBound,
    bounded_loss_gap_from_cmi,
    finite_decision_gap_upper_bound,
    finite_decision_gap_resolution,
    finite_cmi_upper_bound,
)
from .core.finite_policy_gap import (
    FINITE_GAP_LOSS_ROUTER_V1,
    FinitePolicyGapUpperBound,
    finite_coordinate_bernstein_gap_upper_bound,
    finite_policy_gap_resolution,
    finite_policy_gap_upper_bound,
    finite_loss_routed_gap_upper_bound,
)
from .core.declared_modulus import (
    BayesActionJensenDispersion,
    DeclaredCurvaturePoint,
    DeclaredCurvatureProfile,
    DeclaredModulusProfile,
    bayes_action_jensen_dispersion,
    declared_action_curvature_profile,
    declared_feature_modulus_profile,
    feature_barycentric_dispersion,
)
from .core.modulus_calibration import (
    REPEATED_MODULUS_ROUTER_V1,
    CalibratedBarycentricCertificate,
    RepeatedEnvironmentDiscrepancyCalibration,
    calibrate_repeated_environment_discrepancy,
    calibrate_repeated_environment_discrepancy_v1,
    calibrated_barycentric_lower_bound,
    repeated_environment_barycentric_costs,
    support_aware_barycentric_cost_ranges,
)
from .core.residual_information import (
    hierarchical_residual_information,
    hierarchical_residual_ladder,
    multiscale_residual_information,
    support_residual_information,
)
from .core.action_dispersion_calibration import (
    RepeatedActionDispersionCalibration,
    calibrate_repeated_action_dispersion,
)
from .core.multipoint_pinball import (
    MultipointPinballCertificate,
    mixture_dkelly_pinball_portfolio,
    overlapping_pinball_barycentric_portfolio,
)
from .ladder import evaluate_ladder

__all__ = [
    "certify", "Certificate", "ClaimTier", "FoldMatrix", "GuardSpec",
    "Representation", "TaskCard", "DependenceGuardSpec", "RangeDependenceGuardSpec",
    "PairMatrix", "ReplicateGuardSpec",
    "evaluate_ladder", "dependence_profile",
    "RiskInterval", "close_information_order", "information_value_interval",
    "decompose_achieved_gain",
    "RepresentationInterventionAudit", "audit_representation_intervention",
    "LowerEvidence", "GapEvidence", "CompiledCertificate",
    "compile_certificate_graph",
    "EvidenceContext", "lower_evidence_from_result",
    "gap_evidence_from_cap", "gap_evidence_from_profile",
    "gap_evidence_from_finite_cmi",
    "gap_evidence_from_finite_decision",
    "structural_zero_gap",
    "canonical_manifest_hash", "compile_certificate_manifest",
    "compile_certificate_manifest_file",
    "bounded_sample_pairwise_bound", "sharp_pairwise_population_bound",
    "bounded_cluster_pairwise_bound", "bounded_cluster_sensitivity_curve",
    "median_of_means_cluster_pairwise_bound",
    "catoni_cluster_pairwise_bound",
    "empirical_bernstein_patient_pairwise_bound",
    "paired_replicate_lower_bound",
    "paired_replicate_ensemble_sensitivity",
    "paired_replicate_sensitivity_curve",
    "mae_pair_group_envelope", "decision_pair_group_envelope",
    "geometry_routed_decision_envelope",
    "MomentSAAResult", "cantelli_saa_lower_bound",
    "catoni_second_moment_saa_lower_bound",
    "location_saa_score_variance_cap",
    "TruncatedSAACertificate", "dkelly_lower_mean",
    "truncated_saa_portfolio",
    "FiniteCellSAAResult", "finite_cell_saa_lower_bound",
    "FiniteCellLatticeResult", "finite_cell_information_lattice",
    "FiniteRefinementTest", "finite_refinement_permutation_test",
    "FixedPolicyInformationValue", "fixed_policy_information_value_lower_bound",
    "ClusteredFiniteCellSAAResult", "ClusteredFixedPolicyInformationValue",
    "clustered_finite_cell_saa_lower_bound",
    "clustered_fixed_policy_information_value_lower_bound",
    "clustered_block_saa_portfolio",
    "DominatedSAACertificate",
    "dominated_saa_lower_bound",
    "FiniteCMIUpperBound", "finite_cmi_upper_bound",
    "bounded_loss_gap_from_cmi",
    "FiniteDecisionGapUpperBound", "finite_decision_gap_upper_bound",
    "finite_decision_gap_resolution",
    "FinitePolicyGapUpperBound", "finite_policy_gap_resolution",
    "finite_policy_gap_upper_bound",
    "finite_coordinate_bernstein_gap_upper_bound",
    "FINITE_GAP_LOSS_ROUTER_V1", "finite_loss_routed_gap_upper_bound",
    "BayesActionJensenDispersion", "bayes_action_jensen_dispersion",
    "DeclaredCurvaturePoint", "DeclaredCurvatureProfile",
    "declared_action_curvature_profile",
    "DeclaredModulusProfile", "declared_feature_modulus_profile",
    "feature_barycentric_dispersion",
    "RepeatedEnvironmentDiscrepancyCalibration",
    "REPEATED_MODULUS_ROUTER_V1",
    "CalibratedBarycentricCertificate",
    "calibrate_repeated_environment_discrepancy",
    "calibrate_repeated_environment_discrepancy_v1",
    "calibrated_barycentric_lower_bound",
    "repeated_environment_barycentric_costs",
    "support_aware_barycentric_cost_ranges",
    "RepeatedActionDispersionCalibration",
    "calibrate_repeated_action_dispersion",
    "MultipointPinballCertificate",
    "mixture_dkelly_pinball_portfolio",
    "overlapping_pinball_barycentric_portfolio",
    "multiscale_residual_information",
    "support_residual_information",
    "hierarchical_residual_information",
    "hierarchical_residual_ladder",
]
