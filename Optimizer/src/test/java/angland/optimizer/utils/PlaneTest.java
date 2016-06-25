package angland.optimizer.utils;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import angland.optimizer.vec.OrientedPlane;

public class PlaneTest {

  private static final double TOLERANCE = 10e-6;

  @Test
  public void testNormalize() {
    Map<String, Double> initial = new HashMap<>();
    initial.put("x", 3.0);
    initial.put("y", 4.0);
    OrientedPlane<String> normalPlane = new OrientedPlane<>(initial, -10).normalize();
    assertEquals(.6, normalPlane.getVec().get("x"), TOLERANCE);
    assertEquals(.8, normalPlane.getVec().get("y"), TOLERANCE);
    assertEquals(-2, normalPlane.getOffset(), TOLERANCE);
  }
}
