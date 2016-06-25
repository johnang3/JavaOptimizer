package angland.optimizer.optimizer;

import static angland.optimizer.var.Expression.var;
import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.var.Calculation;
import angland.optimizer.var.Expression;
import angland.optimizer.var.IndexedKey;

public class GradientDescentOptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {
    Expression<String> aSquared = var("a").power(2);
    Expression<String> bSquared = var("b").power(2);
    Expression<String> objectiveFunction = aSquared.plus(bSquared);
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("a"), 200.0);
    startingPoint.put(IndexedKey.scalarKey("b"), 200.0);
    Calculation<String> solution =
        GradientDescentOptimizer.stepToMinimum(objectiveFunction, startingPoint, new HashMap<>(),
            10000, 10e-6);
    assertEquals(0.0, solution.value(), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("b")), TOLERANCE);
  }

}
