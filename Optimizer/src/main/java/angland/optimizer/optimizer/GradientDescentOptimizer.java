package angland.optimizer.optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.StreamingSum;
import angland.optimizer.vec.MathUtils;


public class GradientDescentOptimizer {


  /**
   * Returns a new context which is equal to the original minus the normalized gradient. The step
   * distance will no
   * 
   * @return
   */
  public static <Result, VarType> Map<ContextKey<VarType>, Double> step(
      IScalarValue<VarType> calculation, Map<ContextKey<VarType>, Double> context,
      Map<ContextKey<VarType>, Range> variableRanges, double gradientMultiplier) {
    if (gradientMultiplier <= 0) {
      throw new RuntimeException("MaxStepDistance must be greater than 0.");
    }
    Map<ContextKey<VarType>, Double> result = new HashMap<>();
    for (Map.Entry<ContextKey<VarType>, Double> contextEntry : context.entrySet()) {
      if (contextEntry.getValue() == null) {
        throw new RuntimeException("Null value for entry of key " + contextEntry.getKey());
      }
      double stepped =
          contextEntry.getValue() - calculation.d(contextEntry.getKey()) * gradientMultiplier;
      if (Double.isNaN(stepped)) {
        throw new RuntimeException("Stepped to a NaN value.");
      }
      Range range = variableRanges.get(contextEntry.getKey());
      if (range != null) {
        if (stepped < range.getMin()) {
          stepped = range.getMin();
        } else if (stepped > range.getMax()) {
          stepped = range.getMax();
        }
      }

      result.put(contextEntry.getKey(), stepped);
    }
    return result;
  }

  public static <Result, VarType> Map<ContextKey<VarType>, Double> stepNormalized(
      IScalarValue<VarType> calculation, Map<ContextKey<VarType>, Double> context,
      double stepDistance) {
    Map<ContextKey<VarType>, Double> result =
        MathUtils.add(context,
            MathUtils.adjustToMagnitude(calculation.getGradient(), -stepDistance));
    return result;
  }

  public static <Result, VarKey> Solution<Result, VarKey> stepToMinimum(
      Function<Map<ContextKey<VarKey>, Double>, Result> getResult,
      Function<Result, IScalarValue<VarKey>> getObjective,
      Map<ContextKey<VarKey>, Range> variableRanges,
      Map<ContextKey<VarKey>, Double> initialContext, double step, double minStep) {
    Solution<Result, VarKey> bestResult = new Solution<>(initialContext, getResult, getObjective);
    while (step > minStep) {
      Solution<Result, VarKey> next = null;
      while ((next =
          new Solution<>(step(bestResult.getObjective(), bestResult.getContext(), variableRanges,
              step), getResult, getObjective)).getObjective().value() < bestResult.getObjective()
          .value()) {
        bestResult = next;
      }
      step /= 2;
    }
    return bestResult;
  }

  public static <Result, VarKey> Solution<Result, VarKey> normalizedStepToMinimum(
      Function<Map<ContextKey<VarKey>, Double>, Result> getResult,
      Function<Result, IScalarValue<VarKey>> getObjective,
      Map<ContextKey<VarKey>, Double> initialContext, double step, double minStep) {
    Solution<Result, VarKey> bestResult = new Solution<>(initialContext, getResult, getObjective);
    while (step > minStep) {
      Solution<Result, VarKey> next = null;
      while ((next =
          new Solution<>(stepNormalized(bestResult.getObjective(), bestResult.getContext(), step),
              getResult, getObjective)).getObjective().value() < bestResult.getObjective().value()) {
        bestResult = next;
      }
      step /= 2;
    }
    return bestResult;
  }

  private static class ConstrainedSolution<VarKey, Result> {
    private final Result result;
    private final IScalarValue<VarKey> unweightedShortfallSum;

    public ConstrainedSolution(Result result, IScalarValue<VarKey> unweightedShortfalls) {
      super();
      this.result = result;
      this.unweightedShortfallSum = unweightedShortfalls;
    }



  }


  public static <Result, VarKey> Solution<Result, VarKey> optimizeWithConstraints(
      Function<Map<ContextKey<VarKey>, Double>, Result> getResult,
      Function<Result, IScalarValue<VarKey>> getObjective,
      List<Function<Map<ContextKey<VarKey>, Double>, IScalarValue<VarKey>>> zeroMinimumConstraints,
      UnaryOperator<IScalarValue<VarKey>> penaltyScalar,
      Map<ContextKey<VarKey>, Double> initialContext, double step, double minStep,
      double exceedanceTolerance) {
    IScalarValue<VarKey> penaltyMultiplier = IScalarValue.constant(1.0);
    IScalarValue<VarKey> zero = IScalarValue.constant(0);
    IScalarValue<VarKey> minusOne = IScalarValue.constant(-1);
    Map<ContextKey<VarKey>, Double> currentContext = initialContext;
    IScalarValue<VarKey> unweightedPenalty = null;
    Solution<ConstrainedSolution<VarKey, Result>, VarKey> penaltySolution = null;
    do {
      Function<Map<ContextKey<VarKey>, Double>, ConstrainedSolution<VarKey, Result>> getConstrainedSolution =
          ctx -> {
            Result r = getResult.apply(ctx);
            List<IScalarValue<VarKey>> constraintViolations =
                zeroMinimumConstraints.stream().map(c -> IScalarValue.min(c.apply(ctx), zero))
                    .collect(Collectors.toList());
            IScalarValue<VarKey> totalViolation =
                new StreamingSum<>(constraintViolations).cache().times(minusOne);
            return new ConstrainedSolution<>(r, totalViolation);
          };
      IScalarValue<VarKey> penaltyMultiplierFinal = penaltyMultiplier;
      Function<ConstrainedSolution<VarKey, Result>, IScalarValue<VarKey>> getPenalizedObjective =
          cs -> {
            return getObjective.apply(cs.result).plus(
                penaltyScalar.apply(cs.unweightedShortfallSum.times(penaltyMultiplierFinal)));
          };

      penaltySolution =
          normalizedStepToMinimum(getConstrainedSolution, getPenalizedObjective, currentContext,
              step, minStep);
      unweightedPenalty = penaltySolution.getResult().unweightedShortfallSum;
      currentContext = penaltySolution.getContext();
      penaltyMultiplier = penaltyMultiplier.times(IScalarValue.constant(10));
    } while (unweightedPenalty.value() > exceedanceTolerance);
    return new Solution<>(currentContext, penaltySolution.getResult().result,
        penaltySolution.getObjective());
  }
}
