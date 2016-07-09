package angland.optimizer.var;


import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import angland.optimizer.var.scalar.IScalarValue;

public class CalculationTest {

  private static final double TOLERANCE = 10e-6;

  private void testDerivative(double expected, IScalarValue<String> scalar, String key) {
    assertEquals(expected, scalar.d(IndexedKey.scalarKey(key)), TOLERANCE);
    assertEquals(expected, scalar.getGradient().getOrDefault(IndexedKey.scalarKey(key), 0.0),
        TOLERANCE);
  }

  @Test
  public void testConstant() {
    IScalarValue<String> fiveCalced = IScalarValue.constant(5.0);
    assertEquals(5.0, fiveCalced.value(), TOLERANCE);
    testDerivative(0, fiveCalced, "x");
  }

  @Test
  public void testVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    IScalarValue<String> x = IScalarValue.var("x", context);
    assertEquals(5.0, x.value(), TOLERANCE);
    testDerivative(1, x, "x");
  }

  @Test
  public void testAdd() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    IScalarValue<String> result =
        IScalarValue.var("a", context).plus(IScalarValue.var("b", context));
    assertEquals(7.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    testDerivative(1.0, result, "a");
    testDerivative(1.0, result, "b");
  }

  @Test
  public void testSubtract() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    IScalarValue<String> result =
        IScalarValue.var("a", context).minus(IScalarValue.var("b", context));
    assertEquals(-1.0, result.value(), TOLERANCE);
    testDerivative(1.0, result, "a");
    testDerivative(-1.0, result, "b");
  }

  @Test
  public void testVarTimesScalar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    IScalarValue<String> result = IScalarValue.var("x", context).times(IScalarValue.constant(5));
    assertEquals(25.0, result.value(), TOLERANCE);
    testDerivative(5.0, result, "x");
  }

  @Test
  public void testVarTimesVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    context.put(IndexedKey.scalarKey("b"), 3.0);
    IScalarValue<String> result =
        IScalarValue.var("a", context).times(IScalarValue.var("b", context));
    assertEquals(6.0, result.value(), TOLERANCE);
    testDerivative(3.0, result, "a");
    testDerivative(2.0, result, "b");
  }

  @Test
  public void testDivide() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 1.0);
    context.put(IndexedKey.scalarKey("b"), 2.0);
    IScalarValue<String> result =
        IScalarValue.var("a", context).divide(IScalarValue.var("b", context));
    assertEquals(.5, result.value(), TOLERANCE);
    testDerivative(.5, result, "a");
    testDerivative(-.25, result, "b");
  }

  @Test
  public void testVarPower() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    IScalarValue<String> result = IScalarValue.var("a", context).power(2).power(2);
    assertEquals(81.0, result.value(), TOLERANCE);
    testDerivative(108.0, result, "a");
  }


  @Test
  public void testLn() {
    IScalarValue<String> a = IScalarValue.constant(1.0);
    assertEquals(0.0, a.ln().value(), TOLERANCE);
  }

  @Test
  public void testLnDerivative() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .5);
    IScalarValue<String> a = IScalarValue.var("a", context);
    testDerivative(6.0, a.power(3).ln(), "a");
  }

  @Test
  public void testExpDerivative() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    IScalarValue<String> a = IScalarValue.var("a", context);
    testDerivative(5.0 * Math.exp(15), a.times(IScalarValue.constant(5)).exp(), "a");
  }

  @Test
  public void testSigmoid() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    IScalarValue<String> a = IScalarValue.var("a", context);
    IScalarValue<String> sigmoidPointNineA = a.times(IScalarValue.constant(.9)).sigmoid();
    assertEquals(.662622, sigmoidPointNineA.value(), TOLERANCE);
    testDerivative(.201199, sigmoidPointNineA, "a");
  }

  @Test
  public void testTanH() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    IScalarValue<String> a = IScalarValue.var("a", context);
    IScalarValue<String> tanhPointNineA = a.times(IScalarValue.constant(.9)).tanh();
    assertEquals(.588259, tanhPointNineA.value(), TOLERANCE);
    testDerivative(.588556, tanhPointNineA, "a");
  }

  @Test
  public void testZeroEmptiesDerivativeStream() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    IScalarValue<String> a = IScalarValue.var("a", context);
    IScalarValue<String> aTimesZero = a.times(IScalarValue.constant(0));
    AtomicInteger i = new AtomicInteger(0);
    aTimesZero.actOnKeyedDerivatives(x -> i.incrementAndGet());
    assertEquals(0, i.get());
  }
}
