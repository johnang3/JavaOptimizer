package angland.optimizer.var;

import static angland.optimizer.var.Expression.constant;
import static angland.optimizer.var.Expression.var;
import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class CalculationTest {

  private static final double TOLERANCE = 10e-6;

  @Test
  public void testConstant() {
    Expression<String> five = constant(5);
    Calculation<String> fiveCalced = five.evaluate(new HashMap<>());
    assertEquals(5.0, fiveCalced.value(), TOLERANCE);
    assertEquals(0.0, fiveCalced.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testVar() {
    Expression<String> x = var("x");
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    Calculation<String> result = x.evaluate(context);
    assertEquals(5.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testAdd() {
    Expression<String> aPlusB = var("a").plus(var("b"));
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    Calculation<String> result = aPlusB.evaluate(context);
    assertEquals(7.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testSubtract() {
    Expression<String> aPlusB = var("a").minus(var("b"));
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    Calculation<String> result = aPlusB.evaluate(context);
    assertEquals(-1.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(-1.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testVarTimesScalar() {
    Expression<String> fiveX = var("x").times(constant(5));
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    Calculation<String> result = fiveX.evaluate(context);
    assertEquals(25.0, result.value(), TOLERANCE);
    assertEquals(5.0, result.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testVarTimesVar() {
    Expression<String> aTimesB = var("a").times(var("b"));
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    context.put(IndexedKey.scalarKey("b"), 3.0);
    Calculation<String> result = aTimesB.evaluate(context);
    assertEquals(6.0, result.value(), TOLERANCE);
    assertEquals(3.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(2.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testDivide() {
    Expression<String> aOverB = var("a").divide(var("b"));
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 1.0);
    context.put(IndexedKey.scalarKey("b"), 2.0);
    Calculation<String> result = aOverB.evaluate(context);
    assertEquals(.5, result.value(), TOLERANCE);
    assertEquals(.5, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(-.25, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testVarPower() {
    Expression<String> aCubed = var("a").power(3);
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    Calculation<String> result = aCubed.evaluate(context);
    assertEquals(8.0, result.value(), TOLERANCE);
    assertEquals(12.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
  }
}
