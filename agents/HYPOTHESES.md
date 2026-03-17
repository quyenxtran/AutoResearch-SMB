# HYPOTHESES.md - Provisional Beliefs

This document captures emerging patterns and hypotheses that show promise but require further validation before becoming established skills. These represent our current best guesses about system behavior that should be tested in future optimization runs.

## Flow Optimization Hypotheses

### H1: Fdes Purity Impact is Non-Linear

**Hypothesis**: Increasing Fdes has diminishing returns on recovery improvement while purity degradation accelerates at higher flow rates.

**Rationale**: Initial Fdes increases provide significant recovery gains with minimal purity loss, but system approaches saturation where additional desorbent primarily displaces impurities.

**Test Plan**: 
- Systematically vary Fdes from 1.0 to 2.5 mL/min in 0.1 mL increments
- Measure recovery and purity response curves
- Identify inflection point where purity degradation accelerates

**Validation Criteria**: 
- Recovery curve shows diminishing returns (second derivative < 0)
- Purity curve shows accelerating degradation (second derivative < 0)
- Clear inflection point identified with statistical significance

**Current Evidence**: Preliminary data suggests non-linear behavior but insufficient data points for confirmation

**Confidence**: Low-Medium

**Priority**: High

### H2: tstep Optimization is Layout-Dependent

**Hypothesis**: Optimal tstep/Ffeed ratio varies significantly between different NC layouts due to zone volume differences.

**Rationale**: Different layouts have varying zone volumes, requiring different residence times for equivalent separation efficiency.

**Test Plan**:
- For each NC layout, optimize tstep while keeping Ffeed constant
- Calculate optimal tstep/Ffeed ratio for each layout
- Compare ratios across layouts to identify systematic differences

**Validation Criteria**:
- Statistically significant differences in optimal ratios between layouts
- Ratio differences correlate with zone volume ratios
- Layout-specific optimization outperforms universal ratio

**Current Evidence**: Layout (1,2,3,2) appears to prefer longer tstep than (2,2,2,2) but sample size small

**Confidence**: Low

**Priority**: Medium

### H3: F1 Maximum is Limited by Zone 2/3 Interface Stability

**Hypothesis**: F1 cannot be increased beyond a certain point without destabilizing the Zone 2/3 interface, leading to purity collapse.

**Rationale**: High internal circulation rates may disrupt the concentration gradient between adsorption and desorption zones.

**Test Plan**:
- Incrementally increase F1 while monitoring zone interface sharpness
- Identify F1 value where purity begins rapid decline
- Correlate with simulated concentration profiles

**Validation Criteria**:
- Clear threshold F1 value identified where purity drops >5%
- Concentration profile shows interface broadening at threshold
- Threshold correlates with zone dimensions

**Current Evidence**: F1 increases beyond 4.5 mL/min show inconsistent results

**Confidence**: Low

**Priority**: Medium

## Constraint Interaction Hypotheses

### H4: Purity Constraint is More Sensitive to Flow Perturbations than Recovery

**Hypothesis**: Small flow changes have larger impact on purity than on recovery, making purity the limiting constraint in most optimizations.

**Rationale**: Purity requires sharp concentration fronts, which are more easily disrupted than the integrated recovery metric.

**Test Plan**:
- Apply identical flow perturbations and measure sensitivity of each constraint
- Calculate sensitivity coefficients for purity vs recovery
- Test across different operating points

**Validation Criteria**:
- Purity sensitivity coefficient consistently higher than recovery coefficient
- Difference statistically significant across multiple operating points
- Sensitivity ratio > 1.5 in most cases

**Current Evidence**: Observational evidence suggests purity is harder to maintain

**Confidence**: Low

**Priority**: High

### H5: MeOH Concentration is a Leading Indicator of Purity Violations

**Hypothesis**: Extract MeOH concentration begins rising 2-3 optimization steps before purity violations become apparent.

**Rationale**: MeOH acts as a displacing agent; its increasing presence indicates weakening separation efficiency before purity metrics reflect it.

**Test Plan**:
- Monitor MeOH concentration trajectory in optimization sequences leading to purity violations
- Compare to sequences that maintain purity
- Establish early warning threshold

**Validation Criteria**:
- MeOH concentration rise precedes purity violation in >70% of cases
- Predictive threshold identified with <10% false positive rate
- Lead time of 2-3 optimization steps confirmed

**Current Evidence**: Anecdotal observations suggest correlation but no systematic study

**Confidence**: Very Low

**Priority**: Medium

## Fidelity and Discretization Hypotheses

### H6: Low-Fidelity Reliability Degrades Near Constraint Boundaries

**Hypothesis**: Low-fidelity screening becomes unreliable when candidates are close to constraint boundaries due to discretization errors.

**Rationale**: Coarse discretization may not capture the fine-scale transport phenomena that determine constraint satisfaction.

**Test Plan**:
- Compare low vs high fidelity results for candidates with normalized violation < 0.1
- Measure prediction accuracy as function of constraint proximity
- Identify reliability threshold

**Validation Criteria**:
- Prediction accuracy drops significantly for candidates with violation < 0.1
- Clear threshold identified where low-fidelity becomes unreliable
- High-fidelity required within specific violation range

**Current Evidence**: Some evidence of low-fidelity failures near boundaries but not systematically studied

**Confidence**: Low

**Priority**: High

### H7: Optimal Discretization Varies with Flow Regime

**Hypothesis**: Different flow regimes (low vs high F1, different Ffeed/Fdes ratios) require different discretization levels for accurate simulation.

**Rationale**: High flow rates may require finer discretization to capture sharp concentration fronts, while low flows may be adequately modeled with coarse grids.

**Test Plan**:
- Test multiple discretization levels across different flow regimes
- Identify where coarse grids fail to capture key phenomena
- Develop regime-specific discretization rules

**Validation Criteria**:
- Statistically significant differences in results between discretization levels for specific flow regimes
- Clear pattern emerges linking flow characteristics to required discretization
- Regime-specific rules improve prediction accuracy

**Current Evidence**: Observational evidence of discretization sensitivity in high-flow cases

**Confidence**: Very Low

**Priority**: Low

## Layout and Topology Hypotheses

### H8: Zone 1 Fragmentation Improves Purity but Reduces Productivity

**Hypothesis**: Splitting Zone 1 across multiple columns improves separation efficiency but reduces throughput due to flow distribution losses.

**Rationale**: Multiple Zone 1 columns provide better flow distribution and reduced channeling, but introduce additional pressure drops and dead volumes.

**Test Plan**:
- Compare layouts with single vs multiple Zone 1 columns
- Measure purity and productivity for equivalent total Zone 1 volume
- Quantify pressure drop and dead volume effects

**Validation Criteria**:
- Multiple Zone 1 layouts show statistically higher purity
- Productivity reduction quantified and correlated with pressure drop
- Tradeoff curve established for design optimization

**Current Evidence**: Layout (1,2,3,2) vs (2,2,2,2) suggests this pattern but not conclusive

**Confidence**: Low

**Priority**: Medium

### H9: Zone 4 Size has Non-Monotonic Impact on Recovery

**Hypothesis**: Recovery initially improves with Zone 4 size but then decreases due to remixing effects in oversized zones.

**Rationale**: Larger Zone 4 provides more desorption capacity, but excessive size may allow remixing of separated components.

**Test Plan**:
- Systematically vary Zone 4 size while keeping other zones constant
- Measure recovery as function of Zone 4 volume
- Identify optimal Zone 4 size

**Validation Criteria**:
- Recovery curve shows maximum at intermediate Zone 4 sizes
- Remixing evidenced by concentration profile analysis
- Optimal size correlates with desorption kinetics

**Current Evidence**: Limited data suggests non-monotonic behavior but insufficient for confirmation

**Confidence**: Very Low

**Priority**: Low

## Solver and Algorithm Hypotheses

### H10: IPOPT Performance Correlates with Constraint Violation Magnitude

**Hypothesis**: Solver convergence time and success rate are inversely related to the magnitude of initial constraint violations.

**Rationale**: Large violations create complex optimization landscapes that are harder for gradient-based solvers to navigate.

**Test Plan**:
- Measure solver performance metrics across candidates with varying initial violations
- Correlate convergence time and success rate with violation magnitude
- Develop pre-screening rules based on violation thresholds

**Validation Criteria**:
- Strong inverse correlation between violation magnitude and solver performance
- Clear violation threshold identified where solver performance degrades
- Pre-screening rules improve overall optimization efficiency

**Current Evidence**: Observational evidence of solver difficulties with high violations

**Confidence**: Low

**Priority**: Medium

### H11: Warm-Start Benefits Diminish with Layout Changes

**Hypothesis**: Using optimized flows from one layout as starting point for different layout provides minimal benefit due to different flow requirements.

**Rationale**: Different layouts have fundamentally different flow distributions and zone requirements.

**Test Plan**:
- Compare optimization performance with and without warm-start across layout changes
- Measure convergence time and final solution quality
- Quantify warm-start benefit as function of layout similarity

**Validation Criteria**:
- Warm-start provides minimal benefit (<5%) when layout changes
- Benefit correlates with layout similarity metrics
- No warm-start preferred for layout exploration phase

**Current Evidence**: Anecdotal evidence suggests limited warm-start benefit across layouts

**Confidence**: Very Low

**Priority**: Low

## Implementation and Testing Strategy

### Hypothesis Testing Protocol

1. **Design of Experiments**: Each hypothesis requires systematic testing with controlled variables
2. **Statistical Validation**: Results must meet statistical significance thresholds (p < 0.05)
3. **Cross-Validation**: Hypotheses should be tested across multiple feed compositions and operating conditions
4. **Documentation**: All test results must be documented in research.md with full experimental details

### Hypothesis Lifecycle Management

- **Active Testing**: Currently being validated through optimization runs
- **Pending Validation**: Identified but not yet systematically tested
- **Confirmed**: Validated and ready for promotion to SKILLS.md
- **Rejected**: Tested and found invalid, with explanation of why
- **Retired**: No longer applicable due to system changes

### Integration with Optimization Workflow

- **Real-time Testing**: Incorporate hypothesis testing into regular optimization runs
- **Automated Validation**: Develop scripts to automatically test hypotheses during optimization
- **Feedback Loop**: Use hypothesis test results to refine optimization strategies
- **Knowledge Transfer**: Promote confirmed hypotheses to skills, update rejected ones with explanations

### Priority Assignment Criteria

**High Priority**: 
- Direct impact on optimization success rate
- Can be tested with existing infrastructure
- Likely to provide immediate practical benefits

**Medium Priority**:
- Important for understanding system behavior
- Requires some additional experimental setup
- Moderate impact on optimization strategies

**Low Priority**:
- Interesting but not critical for current objectives
- Requires significant additional resources to test
- Long-term research value

---

**Last Updated**: 2024
**Active Hypotheses**: 11
**Next Review**: After next 15 optimization runs
**Testing Status**: H1, H4, H6 in active testing phase