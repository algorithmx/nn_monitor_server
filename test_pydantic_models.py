"""
Comprehensive test suite for pydantic models to identify over-restrictions.
Tests validate against the JSON schema specification in dev-notes/spec.md.
"""

import json
import math
from main import (
    Metadata, IntermediateFeatures, GradientFlow, WeightStats, BiasStats,
    ParameterStatistics, LayerStatistic, CrossLayerAnalysis, MetricsPayload
)
from pydantic import ValidationError


def test_spec_example_payload():
    """Test the exact example payload from the specification."""
    spec_example = {
        "metadata": {
            "run_id": "experiment_2024_0210_v2",
            "timestamp": 1707589200.123,
            "global_step": 1500,
            "batch_size": 64
        },
        "layer_statistics": [
            {
                "layer_id": "encoder.linear1",
                "layer_type": "Linear",
                "depth_index": 0,
                "intermediate_features": {
                    "activation_std": 0.847,
                    "activation_mean": -0.023,
                    "activation_shape": [64, 256],
                    "cross_layer_std_ratio": 0.94
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.152,
                    "gradient_std": 0.0034,
                    "gradient_max_abs": 0.089
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.037,
                        "mean": -0.001,
                        "spectral_norm": 1.42,
                        "frobenius_norm": 2.18
                    },
                    "bias": {
                        "std": 0.012,
                        "mean_abs": 0.008
                    }
                }
            },
            {
                "layer_id": "encoder.linear2",
                "layer_type": "Linear",
                "depth_index": 1,
                "intermediate_features": {
                    "activation_std": 0.795,
                    "activation_mean": 0.015,
                    "activation_shape": [64, 128],
                    "cross_layer_std_ratio": 0.31
                },
                "gradient_flow": {
                    "gradient_l2_norm": 0.089,
                    "gradient_std": 0.0021,
                    "gradient_max_abs": 0.042
                },
                "parameter_statistics": {
                    "weight": {
                        "std": 0.029,
                        "mean": 0.003,
                        "spectral_norm": 1.15,
                        "frobenius_norm": 1.67
                    },
                    "bias": {
                        "std": 0.009,
                        "mean_abs": 0.005
                    }
                }
            }
        ],
        "cross_layer_analysis": {
            "feature_std_gradient": -0.052,
            "gradient_norm_ratio": {
                "encoder.linear2_to_encoder.linear1": 0.586
            }
        }
    }

    try:
        payload = MetricsPayload(**spec_example)
        print("✓ Spec example payload validates successfully")
        return True
    except ValidationError as e:
        print(f"✗ Spec example payload FAILED validation:")
        print(f"  {e}")
        return False


def test_edge_cases():
    """Test edge cases that might be over-restricted."""
    results = []

    # Test 1: Very small activation_std (approaching 0)
    try:
        IntermediateFeatures(
            activation_std=1e-10,
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✓ Very small activation_std (1e-10) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Very small activation_std (1e-10) REJECTED - should be allowed per spec")
        results.append(False)

    # Test 2: activation_std = 0 (could happen with ReLU and all zeros)
    try:
        IntermediateFeatures(
            activation_std=0.0,
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✓ activation_std = 0 accepted")
        results.append(True)
    except ValidationError:
        print("✗ activation_std = 0 REJECTED - gt=0 constraint, but spec doesn't forbid 0")
        results.append(False)

    # Test 3: Large activation_std (>1000)
    try:
        IntermediateFeatures(
            activation_std=5000.0,
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✓ Large activation_std (5000) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large activation_std (5000) REJECTED - le=1000 constraint is too restrictive")
        results.append(False)

    # Test 4: cross_layer_std_ratio = 0 (vanishing gradients case)
    try:
        IntermediateFeatures(
            activation_std=0.5,
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=0.0
        )
        print("✓ cross_layer_std_ratio = 0 accepted")
        results.append(True)
    except ValidationError:
        print("✗ cross_layer_std_ratio = 0 REJECTED - gt=0 constraint, but vanishing case is valid")
        results.append(False)

    # Test 5: Large gradient_l2_norm (>1e6)
    try:
        GradientFlow(
            gradient_l2_norm=5e6,
            gradient_std=1000.0,
            gradient_max_abs=10000.0
        )
        print("✓ Large gradient_l2_norm (5e6) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large gradient_l2_norm (5e6) REJECTED - le=1e6 constraint may be too restrictive")
        results.append(False)

    # Test 6: weight.std = 0 (could happen with zero initialization)
    try:
        WeightStats(
            std=0.0,
            mean=0.0,
            spectral_norm=0.0,
            frobenius_norm=0.0
        )
        print("✓ weight.std = 0 accepted")
        results.append(True)
    except ValidationError:
        print("✗ weight.std = 0 REJECTED - gt=0 constraint, but zero init is valid")
        results.append(False)

    # Test 7: Large weight.std (>100)
    try:
        WeightStats(
            std=500.0,
            mean=0.0,
            spectral_norm=5000.0,
            frobenius_norm=8000.0
        )
        print("✓ Large weight.std (500) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large weight.std (500) REJECTED - le=100 constraint may be too restrictive")
        results.append(False)

    # Test 8: Large spectral_norm (>1e4)
    try:
        WeightStats(
            std=1.0,
            mean=0.0,
            spectral_norm=50000.0,
            frobenius_norm=60000.0
        )
        print("✓ Large spectral_norm (50000) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large spectral_norm (50000) REJECTED - le=1e4 constraint may be too restrictive")
        results.append(False)

    # Test 9: bias.std > 10
    try:
        BiasStats(
            std=50.0,
            mean_abs=25.0
        )
        print("✓ Large bias.std (50) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large bias.std (50) REJECTED - le=10 constraint may be too restrictive")
        results.append(False)

    # Test 10: feature_std_gradient can be any float (including large values)
    try:
        CrossLayerAnalysis(
            feature_std_gradient=1000.0,
            gradient_norm_ratio={}
        )
        print("✓ Large feature_std_gradient (1000) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Large feature_std_gradient (1000) REJECTED - no upper bound in spec")
        results.append(False)

    # Test 11: Non-sequential but sorted depth_index (0, 2 - should be OK now)
    try:
        MetricsPayload(
            metadata=Metadata(
                run_id="test",
                timestamp=1707589200.0,
                global_step=0,
                batch_size=32
            ),
            layer_statistics=[
                LayerStatistic(
                    layer_id="layer1",
                    layer_type="Linear",
                    depth_index=0,
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.5,
                        activation_mean=0.0,
                        activation_shape=[32, 64],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.1,
                        gradient_std=0.01,
                        gradient_max_abs=0.05
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.1,
                            mean=0.0,
                            spectral_norm=1.0,
                            frobenius_norm=1.5
                        ),
                        bias=BiasStats(
                            std=0.01,
                            mean_abs=0.005
                        )
                    )
                ),
                LayerStatistic(
                    layer_id="layer3",
                    layer_type="Linear",
                    depth_index=2,  # Non-sequential but sorted
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.5,
                        activation_mean=0.0,
                        activation_shape=[32, 64],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.1,
                        gradient_std=0.01,
                        gradient_max_abs=0.05
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.1,
                            mean=0.0,
                            spectral_norm=1.0,
                            frobenius_norm=1.5
                        ),
                        bias=BiasStats(
                            std=0.01,
                            mean_abs=0.005
                        )
                    )
                )
            ],
            cross_layer_analysis=CrossLayerAnalysis(
                feature_std_gradient=0.0,
                gradient_norm_ratio={}
            )
        )
        print("✓ Non-sequential but sorted depth_index (0, 2) accepted")
        results.append(True)
    except ValidationError:
        print("✗ Non-sequential but sorted depth_index (0, 2) REJECTED")
        results.append(False)

    # Test 12: Unsorted depth_index (should fail)
    try:
        MetricsPayload(
            metadata=Metadata(
                run_id="test",
                timestamp=1707589200.0,
                global_step=0,
                batch_size=32
            ),
            layer_statistics=[
                LayerStatistic(
                    layer_id="layer2",
                    layer_type="Linear",
                    depth_index=1,  # This comes after layer3 in array but has lower depth_index
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.5,
                        activation_mean=0.0,
                        activation_shape=[32, 64],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.1,
                        gradient_std=0.01,
                        gradient_max_abs=0.05
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.1,
                            mean=0.0,
                            spectral_norm=1.0,
                            frobenius_norm=1.5
                        ),
                        bias=BiasStats(
                            std=0.01,
                            mean_abs=0.005
                        )
                    )
                ),
                LayerStatistic(
                    layer_id="layer3",
                    layer_type="Linear",
                    depth_index=0,  # Unsorted - should fail
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.5,
                        activation_mean=0.0,
                        activation_shape=[32, 64],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.1,
                        gradient_std=0.01,
                        gradient_max_abs=0.05
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.1,
                            mean=0.0,
                            spectral_norm=1.0,
                            frobenius_norm=1.5
                        ),
                        bias=BiasStats(
                            std=0.01,
                            mean_abs=0.005
                        )
                    )
                )
            ],
            cross_layer_analysis=CrossLayerAnalysis(
                feature_std_gradient=0.0,
                gradient_norm_ratio={}
            )
        )
        print("✗ Unsorted depth_index ACCEPTED - should be rejected")
        results.append(False)
    except ValidationError:
        print("✓ Unsorted depth_index rejected (data integrity check)")
        results.append(True)

    return results


def test_negative_values():
    """Test that negative values are properly handled."""
    results = []

    # activation_mean can be negative
    try:
        IntermediateFeatures(
            activation_std=0.5,
            activation_mean=-0.5,  # Negative mean is valid
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✓ Negative activation_mean accepted")
        results.append(True)
    except ValidationError:
        print("✗ Negative activation_mean REJECTED - should be allowed")
        results.append(False)

    # weight.mean can be negative
    try:
        WeightStats(
            std=0.1,
            mean=-0.05,  # Negative mean is valid
            spectral_norm=1.0,
            frobenius_norm=1.5
        )
        print("✓ Negative weight.mean accepted")
        results.append(True)
    except ValidationError:
        print("✗ Negative weight.mean REJECTED - should be allowed")
        results.append(False)

    return results


def test_nan_infinity():
    """Test that NaN and Infinity are properly rejected."""
    results = []

    # NaN should be rejected
    try:
        IntermediateFeatures(
            activation_std=float('nan'),
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✗ NaN in activation_std ACCEPTED - should be rejected")
        results.append(False)
    except ValidationError:
        print("✓ NaN in activation_std rejected")
        results.append(True)

    # Infinity should be rejected
    try:
        IntermediateFeatures(
            activation_std=float('inf'),
            activation_mean=0.0,
            activation_shape=[64, 128],
            cross_layer_std_ratio=None
        )
        print("✗ Infinity in activation_std ACCEPTED - should be rejected")
        results.append(False)
    except ValidationError:
        print("✓ Infinity in activation_std rejected")
        results.append(True)

    return results


def test_optional_bias():
    """Test that bias is optional in ParameterStatistics."""
    results = []

    # Test 1: ParameterStatistics with bias=None (layer without bias)
    try:
        ParameterStatistics(
            weight=WeightStats(
                std=0.1,
                mean=0.0,
                spectral_norm=1.0,
                frobenius_norm=1.5
            ),
            bias=None  # Optional bias - layer like Conv2d with bias=False
        )
        print("✓ ParameterStatistics with bias=None accepted")
        results.append(True)
    except ValidationError:
        print("✗ ParameterStatistics with bias=None REJECTED - bias should be optional")
        results.append(False)

    # Test 2: LayerStatistic with bias=None
    try:
        LayerStatistic(
            layer_id="conv1_no_bias",
            layer_type="Conv2d",
            depth_index=0,
            intermediate_features=IntermediateFeatures(
                activation_std=0.5,
                activation_mean=0.0,
                activation_shape=[32, 64, 28, 28],
                cross_layer_std_ratio=None
            ),
            gradient_flow=GradientFlow(
                gradient_l2_norm=0.1,
                gradient_std=0.01,
                gradient_max_abs=0.05
            ),
            parameter_statistics=ParameterStatistics(
                weight=WeightStats(
                    std=0.1,
                    mean=0.0,
                    spectral_norm=1.0,
                    frobenius_norm=1.5
                ),
                bias=None
            )
        )
        print("✓ LayerStatistic with bias=None accepted")
        results.append(True)
    except ValidationError:
        print("✗ LayerStatistic with bias=None REJECTED - bias should be optional")
        results.append(False)

    # Test 3: Full MetricsPayload with layers having and not having bias
    try:
        MetricsPayload(
            metadata=Metadata(
                run_id="test_optional_bias",
                timestamp=1707589200.0,
                global_step=0,
                batch_size=32
            ),
            layer_statistics=[
                LayerStatistic(
                    layer_id="conv1_with_bias",
                    layer_type="Conv2d",
                    depth_index=0,
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.5,
                        activation_mean=0.0,
                        activation_shape=[32, 64, 28, 28],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.1,
                        gradient_std=0.01,
                        gradient_max_abs=0.05
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.1,
                            mean=0.0,
                            spectral_norm=1.0,
                            frobenius_norm=1.5
                        ),
                        bias=BiasStats(std=0.01, mean_abs=0.005)  # Has bias
                    )
                ),
                LayerStatistic(
                    layer_id="conv2_no_bias",
                    layer_type="Conv2d",
                    depth_index=1,
                    intermediate_features=IntermediateFeatures(
                        activation_std=0.4,
                        activation_mean=0.0,
                        activation_shape=[32, 128, 14, 14],
                        cross_layer_std_ratio=None
                    ),
                    gradient_flow=GradientFlow(
                        gradient_l2_norm=0.08,
                        gradient_std=0.008,
                        gradient_max_abs=0.04
                    ),
                    parameter_statistics=ParameterStatistics(
                        weight=WeightStats(
                            std=0.08,
                            mean=0.0,
                            spectral_norm=1.2,
                            frobenius_norm=1.8
                        ),
                        bias=None  # No bias
                    )
                )
            ],
            cross_layer_analysis=CrossLayerAnalysis(
                feature_std_gradient=-0.1,
                gradient_norm_ratio={}
            )
        )
        print("✓ Full MetricsPayload with mixed bias/none-bias layers accepted")
        results.append(True)
    except ValidationError as e:
        print(f"✗ Full MetricsPayload with mixed bias/none-bias REJECTED: {e}")
        results.append(False)

    return results


def test_run_id_patterns():
    """Test various run_id patterns."""
    results = []

    # Test with hyphen
    try:
        Metadata(
            run_id="my-run-123",
            timestamp=1707589200.0,
            global_step=0,
            batch_size=32
        )
        print("✓ run_id with hyphen accepted")
        results.append(True)
    except ValidationError:
        print("✗ run_id with hyphen REJECTED")
        results.append(False)

    # Test with dot
    try:
        Metadata(
            run_id="my.run.123",
            timestamp=1707589200.0,
            global_step=0,
            batch_size=32
        )
        print("✓ run_id with dot accepted")
        results.append(True)
    except ValidationError:
        print("✗ run_id with dot REJECTED")
        results.append(False)

    # Test with underscore
    try:
        Metadata(
            run_id="my_run_123",
            timestamp=1707589200.0,
            global_step=0,
            batch_size=32
        )
        print("✓ run_id with underscore accepted")
        results.append(True)
    except ValidationError:
        print("✗ run_id with underscore REJECTED")
        results.append(False)

    # Test with colon (common in timestamps) - should now pass
    try:
        Metadata(
            run_id="run:2024:02:10",
            timestamp=1707589200.0,
            global_step=0,
            batch_size=32
        )
        print("✓ run_id with colon accepted")
        results.append(True)
    except ValidationError:
        print("✗ run_id with colon REJECTED")
        results.append(False)

    # Test with slash (e.g., for experiment grouping)
    try:
        Metadata(
            run_id="group1/experiment2",
            timestamp=1707589200.0,
            global_step=0,
            batch_size=32
        )
        print("✓ run_id with slash accepted")
        results.append(True)
    except ValidationError:
        print("✗ run_id with slash REJECTED")
        results.append(False)

    return results


def main():
    print("=" * 70)
    print("PYDANTIC MODEL VALIDATION TEST SUITE")
    print("=" * 70)
    print()

    print("1. Testing spec example payload")
    print("-" * 70)
    spec_result = test_spec_example_payload()
    print()

    print("2. Testing edge cases for over-restrictions")
    print("-" * 70)
    edge_results = test_edge_cases()
    print()

    print("3. Testing negative values")
    print("-" * 70)
    negative_results = test_negative_values()
    print()

    print("4. Testing NaN and Infinity rejection")
    print("-" * 70)
    nan_results = test_nan_infinity()
    print()

    print("5. Testing optional bias")
    print("-" * 70)
    optional_bias_results = test_optional_bias()
    print()

    print("6. Testing run_id patterns")
    print("-" * 70)
    pattern_results = test_run_id_patterns()
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_tests = (
        1 + len(edge_results) + len(negative_results) + len(nan_results) + len(optional_bias_results) + len(pattern_results)
    )
    passed_tests = (
        (1 if spec_result else 0) +
        sum(edge_results) +
        sum(negative_results) +
        sum(nan_results) +
        sum(optional_bias_results) +
        sum(pattern_results)
    )
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Failed: {total_tests - passed_tests}/{total_tests}")

    if not spec_result:
        print()
        print("⚠ CRITICAL: The spec example payload does not validate!")
        print("  This must be fixed before considering the models complete.")

    return all([spec_result] + edge_results + negative_results + nan_results + optional_bias_results + pattern_results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
