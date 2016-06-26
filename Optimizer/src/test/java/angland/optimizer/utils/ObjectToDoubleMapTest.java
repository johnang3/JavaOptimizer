package angland.optimizer.utils;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Ignore;
import org.junit.Test;


public class ObjectToDoubleMapTest {

  @Test
  public void testPutAndGet() {
    ObjectToDoubleMap<String> m = new ObjectToDoubleMap<String>(1, .75);
    m.put("a", 1.0);
    m.put("b", 2.0);
    m.put("c", 3.0);
    m.put("d", 4.0);
    m.put("e", 5.0);
    m.put("a", 6.0);
    m.put("b", 7.0);
    assertEquals(6.0, m.get("a"), 0.0);
    assertEquals(7.0, m.get("b"), 0.0);
    assertEquals(3.0, m.get("c"), 0.0);
    assertEquals(4.0, m.get("d"), 0.0);
    assertEquals(5.0, m.get("e"), 0.0);
  }

  @Test
  public void testCloneWithMultiplier() {
    ObjectToDoubleMap<String> m = new ObjectToDoubleMap<String>(3);
    m.put("a", 10);
    ObjectToDoubleMap<String> tripled = m.cloneWithMultiplier(3);
    assertEquals(tripled.size(), 1);
    assertEquals(30, tripled.get("a"), 0.0);
  }

  @Test
  public void testIncrement() {
    ObjectToDoubleMap<String> m = new ObjectToDoubleMap<>(1, .75);
    for (int i = 0; i < 100; ++i) {
      m.adjust("a", 1.0);
    }
    assertEquals(100.0, m.get("a"), 0.0);
  }

  @Ignore
  @Test
  public void performanceTestRepeatedInsert() {
    long start = System.currentTimeMillis();
    ObjectToDoubleMap<Integer> s = new ObjectToDoubleMap<>(1, .75);
    for (int i = 0; i < 10e8; ++i) {
      s.put(1, 2);
    }
    long end = System.currentTimeMillis();
    System.out.println("ObjectToDouble insert time " + (end - start));
  }

  @Ignore
  @Test
  public void performanceTestRepeatedMerge() {
    long start = System.currentTimeMillis();
    ObjectToDoubleMap<Integer> s = new ObjectToDoubleMap<>(1, .75);
    for (int i = 0; i < 10e8; ++i) {
      s.adjust(1, 2);
    }
    long end = System.currentTimeMillis();
    System.out.println("ObjectToDouble merge time " + (end - start));
  }

  @Ignore
  @Test
  public void hashMapTestRepeatedInsert() {
    long start = System.currentTimeMillis();
    Map<Integer, Double> s = new HashMap<Integer, Double>(1, .75f);
    for (int i = 0; i < 10e8; ++i) {
      s.put(1, 2.0);
    }
    long end = System.currentTimeMillis();
    System.out.println("Hashmap insert time " + (end - start));
  }

  @Ignore
  @Test
  public void hashMapTestRepeatedMerge() {
    long start = System.currentTimeMillis();
    Map<Integer, Double> s = new HashMap<Integer, Double>(1, .75f);
    for (int i = 0; i < 10e8; ++i) {
      s.merge(1, 2.0, Double::sum);
    }
    long end = System.currentTimeMillis();
    System.out.println("Hashmap merge time " + (end - start));
  }


}
