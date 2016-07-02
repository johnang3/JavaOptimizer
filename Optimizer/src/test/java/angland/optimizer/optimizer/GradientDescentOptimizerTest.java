package angland.optimizer.optimizer;

import static angland.optimizer.var.ScalarExpression.var;
import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarExpression;

public class GradientDescentOptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {
    ScalarExpression<String> aSquared = var("a").power(2);
    ScalarExpression<String> bSquared = var("b").power(2);
    ScalarExpression<String> objectiveFunction = aSquared.plus(bSquared);
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("a"), 200.0);
    startingPoint.put(IndexedKey.scalarKey("b"), 200.0);
    Solution<String> solution =
        GradientDescentOptimizer.stepToMinimum(objectiveFunction, startingPoint, new HashMap<>(),
            10000, 10e-6);
    assertEquals(0.0, solution.getResult().value(), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("b")), TOLERANCE);
  }

}
