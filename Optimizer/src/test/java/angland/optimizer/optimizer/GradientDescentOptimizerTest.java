package angland.optimizer.optimizer;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.junit.Ignore;
import org.junit.Test;

import angland.optimizer.var.Context;
import angland.optimizer.var.ContextKey;
import angland.optimizer.var.ContextTemplate;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.scalar.IScalarValue;

public class GradientDescentOptimizerTest {

  private static final double TOLERANCE = 10e-3;

  @Test
  public void testNoConstraints() {
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("a"), 200.0);
    startingPoint.put(IndexedKey.scalarKey("b"), 200.0);
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(startingPoint.keySet().stream().collect(Collectors.toList()));
    Function<Map<ContextKey<String>, Double>, IScalarValue<String>> f = m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> aSquared = IScalarValue.var("a", ctx).power(2);
      IScalarValue<String> bSquared = IScalarValue.var("b", ctx).power(2);
      return aSquared.plus(bSquared);
    };
    Solution<IScalarValue<String>, String> solution =
        GradientDescentOptimizer.stepToMinimum(f, x -> x, new HashMap<>(), contextTemplate
            .createContext(startingPoint).asMap(), 10000, 10e-6);
    assertEquals(0.0, solution.getResult().value(), TOLERANCE);
    assertEquals(0.0,
        solution.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("a"))),
        TOLERANCE);
    assertEquals(0.0,
        solution.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("b"))),
        TOLERANCE);
  }

  @Ignore
  @Test
  public void testConvergingCrossEntropy() {
    Map<IndexedKey<String>, Double> cMap = new HashMap<>();
    Map<ContextKey<String>, Range> ranges = new HashMap<>();
    cMap.put(IndexedKey.scalarKey("a"), .5);
    Context<String> context = ContextTemplate.simpleContext(cMap);
    ranges.put(context.getContextTemplate().getContextKey(IndexedKey.scalarKey("a")), new Range(
        .01, .99));
    for (int i = 0; i < 10; ++i) {
      IScalarValue<String> a = IScalarValue.var("a", context);
      IScalarValue<String> loss = a.ln().times(IScalarValue.constant(-1));
      System.out.println("Val " + a.value());
      System.out.println("Loss " + loss.value());
      context =
          context.getContextTemplate().createContext(
              GradientDescentOptimizer.step(loss, context.asMap(), ranges, 0.1));
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
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(startingPoint.keySet().stream().collect(Collectors.toList()));
    Function<Map<ContextKey<String>, Double>, IScalarValue<String>> getResult = m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> x = IScalarValue.var("x", ctx);
      IScalarValue<String> y = IScalarValue.var("y", ctx);
      return x.power(2.0).times(y.power(3.0)).times(IScalarValue.constant(-1));
    };
    List<Function<Map<ContextKey<String>, Double>, IScalarValue<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> IScalarValue.var("x", contextTemplate.createContext(m)));
    zeroMinimumConstraints.add(m -> IScalarValue.var("y", contextTemplate.createContext(m)));
    zeroMinimumConstraints.add(m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> x = IScalarValue.var("x", ctx);
      IScalarValue<String> y = IScalarValue.var("y", ctx);
      return x.plus(y).minus(IScalarValue.constant(10)).times(IScalarValue.constant(-1));
    });
    Map<ContextKey<String>, Double> initialPoint =
        contextTemplate.createContext(startingPoint).asMap();
    Solution<IScalarValue<String>, String> result =
        GradientDescentOptimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints,
            IScalarValue::exp, initialPoint, 1.0, .00001, .00001);
    assertEquals(4.0,
        result.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("x"))),
        TOLERANCE);
    assertEquals(6.0,
        result.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("y"))),
        TOLERANCE);
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
    ContextTemplate<String> contextTemplate =
        new ContextTemplate<>(startingPoint.keySet().stream().collect(Collectors.toList()));
    Function<Map<ContextKey<String>, Double>, IScalarValue<String>> getResult = m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> x = IScalarValue.var("x", ctx);
      IScalarValue<String> y = IScalarValue.var("y", ctx);
      return x.plus(y).times(IScalarValue.constant(-1));
    };
    List<Function<Map<ContextKey<String>, Double>, IScalarValue<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> IScalarValue.var("x", contextTemplate.createContext(m)));
    zeroMinimumConstraints.add(m -> IScalarValue.var("y", contextTemplate.createContext(m)));
    zeroMinimumConstraints.add(m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> x = IScalarValue.var("x", ctx);
      IScalarValue<String> y = IScalarValue.var("y", ctx);
      return x.plus(y.times(IScalarValue.constant(.5))).minus(IScalarValue.constant(3))
          .times(IScalarValue.constant(-1));
    });
    zeroMinimumConstraints.add(m -> {
      Context<String> ctx = contextTemplate.createContext(m);
      IScalarValue<String> x = IScalarValue.var("x", ctx);
      IScalarValue<String> y = IScalarValue.var("y", ctx);
      return x.times(IScalarValue.constant(.5)).plus(y).minus(IScalarValue.constant(3))
          .times(IScalarValue.constant(-1));
    });
    Map<ContextKey<String>, Double> initialPoint =
        contextTemplate.createContext(startingPoint).asMap();
    Solution<IScalarValue<String>, String> result =
        GradientDescentOptimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints,
            IScalarValue::exp, initialPoint, 1.0, .00001, .00001);
    assertEquals(2.0,
        result.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("x"))),
        TOLERANCE);
    assertEquals(2.0,
        result.getContext().get(contextTemplate.getContextKey(IndexedKey.scalarKey("y"))),
        TOLERANCE);
  }
}
