package angland.optimizer.optimizer;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import org.junit.Test;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;

public class GradientDescentOptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {

    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("a"), 200.0);
    startingPoint.put(IndexedKey.scalarKey("b"), 200.0);
    Function<Map<IndexedKey<String>, Double>, ScalarValue<String>> f = ctx -> {
      ScalarValue<String> aSquared = ScalarValue.var("a", ctx).power(2);
      ScalarValue<String> bSquared = ScalarValue.var("b", ctx).power(2);
      return aSquared.plus(bSquared);
    };
    Solution<ScalarValue<String>, String> solution =
        GradientDescentOptimizer.stepToMinimum(f, x -> x, new HashMap<>(), startingPoint, 10000,
            10e-6);
    assertEquals(0.0, solution.getResult().value(), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("b")), TOLERANCE);
  }

}
