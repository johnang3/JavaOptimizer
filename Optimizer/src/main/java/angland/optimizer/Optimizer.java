package angland.optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.Scalar;
import angland.optimizer.var.scalar.StreamingSum;
import angland.optimizer.vec.MathUtils;


public class Optimizer {


  /**
   * Returns a new context which is equal to the original minus the normalized gradient. The step
   * distance will no
   * 
   * @return
   */
  public static <Result, VarType> Map<IndexedKey<VarType>, Double> step(
      Scalar<VarType> calculation, Map<IndexedKey<VarType>, Double> context,
      Map<IndexedKey<VarType>, Range> variableRanges, double gradientMultiplier) {
    if (gradientMultiplier <= 0) {
      throw new RuntimeException("MaxStepDistance must be greater than 0.");
    }
    Map<IndexedKey<VarType>, Double> result = new HashMap<>();
    for (Map.Entry<IndexedKey<VarType>, Double> contextEntry : context.entrySet()) {
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

  public static <Result, VarType> Map<IndexedKey<VarType>, Double> stepNormalized(
      Scalar<VarType> calculation, Map<IndexedKey<VarType>, Double> context, double stepDistance) {
    Map<IndexedKey<VarType>, Double> result =
        MathUtils.add(context,
            MathUtils.adjustToMagnitude(calculation.getGradient(), -stepDistance));
    return result;
  }

  public static <Result, VarKey> Solution<Result, VarKey> stepToMinimum(
      Function<Map<IndexedKey<VarKey>, Double>, Result> getResult,
      Function<Result, Scalar<VarKey>> getObjective, Map<IndexedKey<VarKey>, Range> variableRanges,
      Map<IndexedKey<VarKey>, Double> initialContext, double step, double minStep) {
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
      Function<Map<IndexedKey<VarKey>, Double>, Result> getResult,
      Function<Result, Scalar<VarKey>> getObjective,
      Map<IndexedKey<VarKey>, Double> initialContext, double step, double minStep) {
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
    private final Scalar<VarKey> unweightedShortfallSum;

    public ConstrainedSolution(Result result, Scalar<VarKey> unweightedShortfalls) {
      super();
      this.result = result;
      this.unweightedShortfallSum = unweightedShortfalls;
    }



  }


  public static <Result, VarKey> Solution<Result, VarKey> optimizeWithConstraints(
      Function<Map<IndexedKey<VarKey>, Double>, Result> getResult,
      Function<Result, Scalar<VarKey>> getObjective,
      List<Function<Map<IndexedKey<VarKey>, Double>, Scalar<VarKey>>> zeroMinimumConstraints,
      UnaryOperator<Scalar<VarKey>> penaltyScalar, Map<IndexedKey<VarKey>, Double> initialContext,
      double step, double minStep, double exceedanceTolerance) {
    Scalar<VarKey> penaltyMultiplier = Scalar.constant(1.0);
    Scalar<VarKey> zero = Scalar.constant(0);
    Scalar<VarKey> minusOne = Scalar.constant(-1);
    Map<IndexedKey<VarKey>, Double> currentContext = initialContext;
    Scalar<VarKey> unweightedPenalty = null;
    Solution<ConstrainedSolution<VarKey, Result>, VarKey> penaltySolution = null;
    do {
      Function<Map<IndexedKey<VarKey>, Double>, ConstrainedSolution<VarKey, Result>> getConstrainedSolution =
          ctx -> {
            Result r = getResult.apply(ctx);
            List<Scalar<VarKey>> constraintViolations =
                zeroMinimumConstraints.stream().map(c -> Scalar.min(c.apply(ctx), zero))
                    .collect(Collectors.toList());
            Scalar<VarKey> totalViolation =
                new StreamingSum<>(constraintViolations).cache().times(minusOne);
            return new ConstrainedSolution<>(r, totalViolation);
          };
      Scalar<VarKey> penaltyMultiplierFinal = penaltyMultiplier;
      Function<ConstrainedSolution<VarKey, Result>, Scalar<VarKey>> getPenalizedObjective =
          cs -> {
            return getObjective.apply(cs.result).plus(
                penaltyScalar.apply(cs.unweightedShortfallSum.times(penaltyMultiplierFinal)));
          };

      penaltySolution =
          normalizedStepToMinimum(getConstrainedSolution, getPenalizedObjective, currentContext,
              step, minStep);
      unweightedPenalty = penaltySolution.getResult().unweightedShortfallSum;
      currentContext = penaltySolution.getContext();
      penaltyMultiplier = penaltyMultiplier.times(Scalar.constant(10));
    } while (unweightedPenalty.value() > exceedanceTolerance);
    return new Solution<>(currentContext, penaltySolution.getResult().result,
        penaltySolution.getObjective());
  }
}
