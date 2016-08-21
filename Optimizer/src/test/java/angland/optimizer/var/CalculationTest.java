package angland.optimizer.var;


import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import angland.optimizer.var.scalar.Scalar;

public class CalculationTest {

  private static final double TOLERANCE = 10e-6;

  private void testDerivative(double expected, Scalar<String> scalar, String key,
      Context<String> context) {
    ContextKey<String> cKey = context.getContextTemplate().getContextKey(IndexedKey.scalarKey(key));
    assertEquals(expected, scalar.d(cKey), TOLERANCE);
    assertEquals(expected, scalar.getGradient().getOrDefault(cKey, 0.0), TOLERANCE);
    assertEquals(expected, scalar.arrayCache(context).getGradient().getOrDefault(cKey, 0.0),
        TOLERANCE);
  }

  @Test
  public void testConstant() {
    Scalar<String> fiveCalced = Scalar.constant(5.0);
    assertEquals(5.0, fiveCalced.value(), TOLERANCE);
    testDerivative(0, fiveCalced, "x", ContextTemplate.simpleContext(new HashMap<>()));
  }

  @Test
  public void testVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> x = Scalar.var("x", ctx);
    assertEquals(5.0, x.value(), TOLERANCE);
    testDerivative(1, x, "x", ctx);
  }

  @Test
  public void testAdd() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("a", ctx).plus(Scalar.var("b", ctx));
    assertEquals(7.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(ctx.getContextTemplate().getContextKey(IndexedKey.scalarKey("a"))),
        TOLERANCE);
    testDerivative(1.0, result, "a", ctx);
    testDerivative(1.0, result, "b", ctx);
  }

  @Test
  public void testSubtract() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("a", ctx).minus(Scalar.var("b", ctx));
    assertEquals(-1.0, result.value(), TOLERANCE);
    testDerivative(1.0, result, "a", ctx);
    testDerivative(-1.0, result, "b", ctx);
  }

  @Test
  public void testVarTimesScalar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("x", ctx).times(Scalar.constant(5));
    assertEquals(25.0, result.value(), TOLERANCE);
    testDerivative(5.0, result, "x", ctx);
  }

  @Test
  public void testVarTimesVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    context.put(IndexedKey.scalarKey("b"), 3.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("a", ctx).times(Scalar.var("b", ctx));
    assertEquals(6.0, result.value(), TOLERANCE);
    testDerivative(3.0, result, "a", ctx);
    testDerivative(2.0, result, "b", ctx);
  }

  @Test
  public void testDivide() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 1.0);
    context.put(IndexedKey.scalarKey("b"), 2.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("a", ctx).divide(Scalar.var("b", ctx));
    assertEquals(.5, result.value(), TOLERANCE);
    testDerivative(.5, result, "a", ctx);
    testDerivative(-.25, result, "b", ctx);
  }

  @Test
  public void testVarPower() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> result = Scalar.var("a", ctx).power(2).power(2);
    assertEquals(81.0, result.value(), TOLERANCE);
    testDerivative(108.0, result, "a", ctx);
  }


  @Test
  public void testLn() {
    Scalar<String> a = Scalar.constant(1.0);
    assertEquals(0.0, a.ln().value(), TOLERANCE);
  }

  @Test
  public void testLnDerivative() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .5);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> a = Scalar.var("a", ctx);
    testDerivative(6.0, a.power(3).ln(), "a", ctx);
  }

  @Test
  public void testExpDerivative() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> a = Scalar.var("a", ctx);
    testDerivative(5.0 * Math.exp(15), a.times(Scalar.constant(5)).exp(), "a", ctx);
  }

  @Test
  public void testSigmoid() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> a = Scalar.var("a", ctx);
    Scalar<String> sigmoidPointNineA = a.times(Scalar.constant(.9)).sigmoid();
    assertEquals(.662622, sigmoidPointNineA.value(), TOLERANCE);
    testDerivative(.201199, sigmoidPointNineA, "a", ctx);
  }

  @Test
  public void testTanH() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    Context<String> ctx = ContextTemplate.simpleContext(context);
    Scalar<String> a = Scalar.var("a", ctx);
    Scalar<String> tanhPointNineA = a.times(Scalar.constant(.9)).tanh();
    assertEquals(.588259, tanhPointNineA.value(), TOLERANCE);
    testDerivative(.588556, tanhPointNineA, "a", ctx);
  }

  @Test
  public void testZeroEmptiesDerivativeStream() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Scalar<String> a = Scalar.var("a", ContextTemplate.simpleContext(context));
    Scalar<String> aTimesZero = a.times(Scalar.constant(0));
    AtomicInteger i = new AtomicInteger(0);
    aTimesZero.actOnKeyedDerivatives(x -> i.incrementAndGet());
    assertEquals(0, i.get());
  }
}
