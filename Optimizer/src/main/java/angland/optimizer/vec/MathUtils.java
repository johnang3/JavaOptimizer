package angland.optimizer.vec;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

/**
 * 
 * @author John Angland
 *
 */
public class MathUtils {

  /**
   * Computes the sum of two map representations of sparse vectors.
   * 
   * @return
   */
  public static <Key> Map<Key, Double> add(Map<Key, Double> a, Map<Key, Double> b) {
    Map<Key, Double> result = new HashMap<>(a.size() + b.size(), 1);
    BiConsumer<Key, Double> accumulator = (k, v) -> result.merge(k, v, Double::sum);
    a.forEach(accumulator);
    b.forEach(accumulator);
    return result;
  }

  /**
   * Returns the result of a-b, where a and b are map representations of vectors.
   * 
   * @param a
   * @param b
   * @return
   */
  public static <Key> Map<Key, Double> subtract(Map<Key, Double> a, Map<Key, Double> b) {
    return add(a, multiply(-1, b));
  }

  /**
   * Multiply a map representing a sparse representation of a vector by a constant scalar.
   * 
   * @return
   */
  public static <Key> Map<Key, Double> multiply(double scalar, Map<Key, Double> vec) {
    Map<Key, Double> result = new HashMap<>(vec.size(), 1);
    vec.forEach((k, v) -> result.put(k, v * scalar));
    return result;
  }

  /**
   * Computes the dot product of two vectors.
   * 
   * @return
   */
  public static <Key> double dot(Map<Key, Double> left, Map<Key, Double> right) {
    Set<Key> output = new HashSet<>(left.size() + right.size(), 1);
    output.addAll(left.keySet());
    output.addAll(right.keySet());
    return output.stream().collect(
        Collectors.summingDouble(k -> left.getOrDefault(k, 0.0) * right.getOrDefault(k, 0.0)));
  }

  /**
   * Returns the l2 norm of the given vector.
   * 
   * @param m
   * @return
   */
  public static double l2Norm(Map<?, Double> m) {
    return Math.sqrt(m.values().stream().collect(Collectors.summingDouble(x -> x * x)));
  }

  public static <Key> Map<Key, Double> adjustToMagnitude(Map<Key, Double> vec, double newNorm) {
    double currentMagnitude = l2Norm(vec);
    return multiply(newNorm / currentMagnitude, vec);
  }


  public static double sigmoid(double x) {
    return 1.0 / (1 + Math.exp(-x));
  }
}
