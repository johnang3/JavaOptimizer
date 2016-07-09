package angland.optimizer.optimizer;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;


public class GradientDescentOptimizer {


  /**
   * Returns a new context which is equal to the original minus the normalized gradient. The step
   * distance will no
   * 
   * @return
   */
  public static <Result, VarType> Map<IndexedKey<VarType>, Double> step(
      IScalarValue<VarType> calculation, Map<IndexedKey<VarType>, Double> context,
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


  public static <Result, VarKey> Solution<Result, VarKey> stepToMinimum(
      Function<Map<IndexedKey<VarKey>, Double>, Result> getResult,
      Function<Result, IScalarValue<VarKey>> getObjective,
      Map<IndexedKey<VarKey>, Range> variableRanges,
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
}
