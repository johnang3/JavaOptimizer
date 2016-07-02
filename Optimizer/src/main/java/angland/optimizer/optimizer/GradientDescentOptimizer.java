package angland.optimizer.optimizer;

import java.util.HashMap;
import java.util.Map;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarExpression;
import angland.optimizer.var.ScalarValue;


public class GradientDescentOptimizer {


  /**
   * Returns a new context which is equal to the original minus the normalized gradient. The step
   * distance will no
   * 
   * @return
   */
  public static <VarType> Map<IndexedKey<VarType>, Double> step(ScalarValue<VarType> calculation,
      Map<IndexedKey<VarType>, Double> context, Map<VarType, Range> variableRanges,
      double gradientMultiplier) {
    if (gradientMultiplier <= 0) {
      throw new RuntimeException("MaxStepDistance must be greater than 0.");
    }
    Map<IndexedKey<VarType>, Double> result = new HashMap<>();
    for (Map.Entry<IndexedKey<VarType>, Double> contextEntry : context.entrySet()) {
      double stepped =
          contextEntry.getValue() - calculation.getGradient().get(contextEntry.getKey())
              * gradientMultiplier;
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

  public static <VarType> Solution<VarType> stepToMinimum(ScalarExpression<VarType> expression,
      Map<IndexedKey<VarType>, Double> initialContext, Map<VarType, Range> variableRanges,
      double step, double minStep) {
    Map<IndexedKey<VarType>, Double> bestContext = initialContext;
    ScalarValue<VarType> best = expression.evaluate(initialContext);
    while (step > minStep) {
      ScalarValue<VarType> next = null;
      Map<IndexedKey<VarType>, Double> nextContext = null;
      while ((next =
          expression.evaluate((nextContext = step(best, bestContext, variableRanges, step))))
          .value() < best.value()) {
        best = next;
        bestContext = nextContext;
      }
      step /= 2;
    }
    return new Solution<>(bestContext, best);
  }
}
