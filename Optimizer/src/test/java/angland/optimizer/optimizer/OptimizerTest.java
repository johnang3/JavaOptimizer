package angland.optimizer.optimizer;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.junit.Ignore;
import org.junit.Test;

import angland.optimizer.Optimizer;
import angland.optimizer.Range;
import angland.optimizer.Solution;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.Scalar;

public class OptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("a"), 200.0);
    startingPoint.put(IndexedKey.scalarKey("b"), 200.0);
    Function<Map<IndexedKey<String>, Double>, Scalar<String>> f = m -> {
      Scalar<String> aSquared = Scalar.var("a", m).power(2);
      Scalar<String> bSquared = Scalar.var("b", m).power(2);
      return aSquared.plus(bSquared);
    };
    Solution<Scalar<String>, String> solution =
        Optimizer.stepToMinimum(f, x -> x, new HashMap<>(), startingPoint, 10000, 10e-6);
    assertEquals(0.0, solution.getResult().value(), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(0.0, solution.getContext().get(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Ignore
  @Test
  public void testConvergingCrossEntropy() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    Map<IndexedKey<String>, Range> ranges = new HashMap<>();
    ranges.put(IndexedKey.scalarKey("a"), new Range(.01, .99));
    for (int i = 0; i < 10; ++i) {
      Scalar<String> a = Scalar.var("a", context);
      Scalar<String> loss = a.ln().times(Scalar.constant(-1));
      System.out.println("Val " + a.value());
      System.out.println("Loss " + loss.value());
      context = Optimizer.step(loss, context, ranges, 0.1);
    }
  }


  /**
   * Verify that we can solve a simple nonlinear program with linear constraints.
   * 
   * Minimize -x^2*y^3 subject to x>=0, y>=0, -x-y+10>=0;
   * 
   * L = -x^2*y^3 + lambda(-x-y+10) + L2(x)
   * 
   * dL/dx = -2x*y^3 - lambda dL/dy = -3x^2*y^2 - lambda
   * 
   * -2x * y^3 = -3x^2 * y^2 -2 * y = -3 * x y = 1.5 x
   * 
   * -x - 1.5x + 10 >= 0
   * 
   * x = 4 y = 6
   * 
   */
  @Test
  public void testMinimizeNonlinearWithLinearConstraint() {
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("x"), 0.0);
    startingPoint.put(IndexedKey.scalarKey("y"), 0.0);
    Function<Map<IndexedKey<String>, Double>, Scalar<String>> getResult = m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.power(2.0).times(y.power(3.0)).times(Scalar.constant(-1));
    };
    List<Function<Map<IndexedKey<String>, Double>, Scalar<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> Scalar.var("x", m));
    zeroMinimumConstraints.add(m -> Scalar.var("y", m));
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y).minus(Scalar.constant(10)).times(Scalar.constant(-1));
    });
    Solution<Scalar<String>, String> result =
        Optimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints, Scalar::exp,
            startingPoint, 1.0, .00001, .00001);
    assertEquals(4.0, result.getContext().get(IndexedKey.scalarKey("x")), TOLERANCE);
    assertEquals(6.0, result.getContext().get(IndexedKey.scalarKey("y")), TOLERANCE);
  }


  /**
   * Confirm that we can solve a simple linear program.
   * 
   * Maximimize x+y subject to x >= 0, y >= 0, x+.5y <= 3; .5x+y <= 3
   */
  @Test
  public void testLinearProgram() {
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("x"), 0.0);
    startingPoint.put(IndexedKey.scalarKey("y"), 0.0);
    Function<Map<IndexedKey<String>, Double>, Scalar<String>> getResult = m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y).times(Scalar.constant(-1));
    };
    List<Function<Map<IndexedKey<String>, Double>, Scalar<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> Scalar.var("x", m));
    zeroMinimumConstraints.add(m -> Scalar.var("y", m));
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y.times(Scalar.constant(.5))).minus(Scalar.constant(3))
          .times(Scalar.constant(-1));
    });
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.times(Scalar.constant(.5)).plus(y).minus(Scalar.constant(3))
          .times(Scalar.constant(-1));
    });

    Solution<Scalar<String>, String> result =
        Optimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints, Scalar::exp,
            startingPoint, 1.0, .00001, .00001);
    assertEquals(2.0, result.getContext().get(IndexedKey.scalarKey("x")), TOLERANCE);
    assertEquals(2.0, result.getContext().get(IndexedKey.scalarKey("y")), TOLERANCE);
  }
}
