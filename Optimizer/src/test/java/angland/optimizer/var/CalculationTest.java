package angland.optimizer.var;


import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Test;

import angland.optimizer.var.scalar.Scalar;

public class CalculationTest {

  private static final double TOLERANCE = 10e-6;

  private void testDerivative(double expected, Scalar<String> scalar, String key) {
    IndexedKey<String> cKey = IndexedKey.scalarKey(key);
    assertEquals(expected, scalar.d(cKey), TOLERANCE);
    assertEquals(expected, scalar.getGradient().getOrDefault(cKey, 0.0), TOLERANCE);
  }

  @Test
  public void testConstant() {
    Scalar<String> fiveCalced = Scalar.constant(5.0);
    assertEquals(5.0, fiveCalced.value(), TOLERANCE);
    testDerivative(0, fiveCalced, "x");
  }

  @Test
  public void testVar() {
    Scalar<String> x = Scalar.var("x", 5.0);
    assertEquals(5.0, x.value(), TOLERANCE);
    testDerivative(1, x, "x");
  }

  @Test
  public void testAdd() {
    Scalar<String> result = Scalar.var("a", 3.0).plus(Scalar.var("b", 4.0));
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
    Scalar<String> result = Scalar.var("a", context).minus(Scalar.var("b", context));
    assertEquals(-1.0, result.value(), TOLERANCE);
    testDerivative(1.0, result, "a");
    testDerivative(-1.0, result, "b");
  }

  @Test
  public void testVarTimesScalar() {
    Scalar<String> result = Scalar.var("x", 5).times(Scalar.constant(5));
    assertEquals(25.0, result.value(), TOLERANCE);
    testDerivative(5.0, result, "x");
  }

  @Test
  public void testVarTimesVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    context.put(IndexedKey.scalarKey("b"), 3.0);
    Scalar<String> result = Scalar.var("a", context).times(Scalar.var("b", context));
    assertEquals(6.0, result.value(), TOLERANCE);
    testDerivative(3.0, result, "a");
    testDerivative(2.0, result, "b");
  }

  @Test
  public void testDivide() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 1.0);
    context.put(IndexedKey.scalarKey("b"), 2.0);
    Scalar<String> result = Scalar.var("a", context).divide(Scalar.var("b", context));
    assertEquals(.5, result.value(), TOLERANCE);
    testDerivative(.5, result, "a");
    testDerivative(-.25, result, "b");
  }

  @Test
  public void testVarPower() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Scalar<String> result = Scalar.var("a", context).power(2).power(2);
    assertEquals(81.0, result.value(), TOLERANCE);
    testDerivative(108.0, result, "a");
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
    Scalar<String> a = Scalar.var("a", context);
    testDerivative(6.0, a.power(3).ln(), "a");
  }

  @Test
  public void testExpDerivative() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Scalar<String> a = Scalar.var("a", context);
    testDerivative(5.0 * Math.exp(15), a.times(Scalar.constant(5)).exp(), "a");
  }

  @Test
  public void testSigmoid() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    Scalar<String> a = Scalar.var("a", context);
    Scalar<String> sigmoidPointNineA = a.times(Scalar.constant(.9)).sigmoid();
    assertEquals(.662622, sigmoidPointNineA.value(), TOLERANCE);
    testDerivative(.201199, sigmoidPointNineA, "a");
  }

  @Test
  public void testTanH() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), .75);
    Scalar<String> a = Scalar.var("a", context);
    Scalar<String> tanhPointNineA = a.times(Scalar.constant(.9)).tanh();
    assertEquals(.588259, tanhPointNineA.value(), TOLERANCE);
    testDerivative(.588556, tanhPointNineA, "a");
  }

  @Test
  public void testZeroEmptiesDerivativeStream() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    Scalar<String> a = Scalar.var("a", context);
    Scalar<String> aTimesZero = a.times(Scalar.constant(0));
    AtomicInteger i = new AtomicInteger(0);
    aTimesZero.actOnKeyedDerivatives(x -> i.incrementAndGet());
    assertEquals(0, i.get());
  }
}
