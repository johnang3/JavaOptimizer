package angland.optimizer.optimizer;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

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
}
