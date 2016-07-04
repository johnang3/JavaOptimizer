package angland.optimizer.var;


import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class CalculationTest {

  private static final double TOLERANCE = 10e-6;

  @Test
  public void testConstant() {
    ScalarValue<String> fiveCalced = ScalarValue.constant(5.0);
    assertEquals(5.0, fiveCalced.value(), TOLERANCE);
    assertEquals(0.0, fiveCalced.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    ScalarValue<String> x = ScalarValue.var("x", context);
    assertEquals(5.0, x.value(), TOLERANCE);
    assertEquals(1.0, x.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testAdd() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    ScalarValue<String> result = ScalarValue.var("a", context).plus(ScalarValue.var("b", context));
    assertEquals(7.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testSubtract() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 3.0);
    context.put(IndexedKey.scalarKey("b"), 4.0);
    ScalarValue<String> result = ScalarValue.var("a", context).minus(ScalarValue.var("b", context));
    assertEquals(-1.0, result.value(), TOLERANCE);
    assertEquals(1.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(-1.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testVarTimesScalar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("x"), 5.0);
    ScalarValue<String> result = ScalarValue.var("x", context).times(ScalarValue.constant(5));
    assertEquals(25.0, result.value(), TOLERANCE);
    assertEquals(5.0, result.d(IndexedKey.scalarKey("x")), TOLERANCE);
  }

  @Test
  public void testVarTimesVar() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    context.put(IndexedKey.scalarKey("b"), 3.0);
    ScalarValue<String> result = ScalarValue.var("a", context).times(ScalarValue.var("b", context));
    assertEquals(6.0, result.value(), TOLERANCE);
    assertEquals(3.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(2.0, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testDivide() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 1.0);
    context.put(IndexedKey.scalarKey("b"), 2.0);
    ScalarValue<String> result =
        ScalarValue.var("a", context).divide(ScalarValue.var("b", context));
    assertEquals(.5, result.value(), TOLERANCE);
    assertEquals(.5, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
    assertEquals(-.25, result.d(IndexedKey.scalarKey("b")), TOLERANCE);
  }

  @Test
  public void testVarPower() {
    Map<IndexedKey<String>, Double> context = new HashMap<>();
    context.put(IndexedKey.scalarKey("a"), 2.0);
    ScalarValue<String> result = ScalarValue.var("a", context).power(3);
    assertEquals(8.0, result.value(), TOLERANCE);
    assertEquals(12.0, result.d(IndexedKey.scalarKey("a")), TOLERANCE);
  }


}
