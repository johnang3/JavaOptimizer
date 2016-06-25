package angland.optimizer.utils;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import angland.optimizer.vec.MathUtils;

public class MathUtilsTest {

  private static final double TOLERANCE = 10e-6;

  private Map<String, Double> x;
  private Map<String, Double> y;
  private Map<String, Double> z;

  @Before
  public void prepareVariables() {
    x = new HashMap<>();
    x.put("a", 2.0);
    x.put("b", 3.0);
    y = new HashMap<>();
    y.put("b", 4.0);
    y.put("c", 5.0);
    z = new HashMap<>();
    z.put("a", 3.0);
    z.put("b", 4.0);
  }


  @Test
  public void testAdd() {
    Map<String, Double> result = MathUtils.add(x, y);
    assertEquals(2.0, result.get("a"), TOLERANCE);
    assertEquals(7.0, result.get("b"), TOLERANCE);
    assertEquals(5.0, result.get("c"), TOLERANCE);
  }

  @Test
  public void testSubtract() {
    Map<String, Double> result = MathUtils.subtract(x, y);
    assertEquals(2.0, result.get("a"), TOLERANCE);
    assertEquals(-1.0, result.get("b"), TOLERANCE);
    assertEquals(-5.0, result.get("c"), TOLERANCE);
  }

  @Test
  public void testScalarTimesVec() {
    Map<String, Double> result = MathUtils.multiply(3.0, x);
    assertEquals(6.0, result.get("a"), TOLERANCE);
    assertEquals(9.0, result.get("b"), TOLERANCE);
  }

  @Test
  public void testDotProduct() {
    assertEquals(12.0, MathUtils.dot(x, y), TOLERANCE);
  }

  @Test
  public void testL2Norm() {
    assertEquals(5.0, MathUtils.l2Norm(z), TOLERANCE);
  }

  @Test
  public void testAdjustedToMagnitude() {
    Map<String, Double> adjusted = MathUtils.adjustToMagnitude(z, 10.0);
    assertEquals(6.0, adjusted.get("a"), TOLERANCE);
    assertEquals(8.0, adjusted.get("b"), TOLERANCE);
  }



}
