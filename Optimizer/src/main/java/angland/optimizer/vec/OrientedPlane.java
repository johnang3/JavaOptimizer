package angland.optimizer.vec;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * 
 * @author John Angland
 *
 * @param <VarType>
 */
public class OrientedPlane<VarType> {

  private final Map<VarType, Double> vec;
  private final double offset;

  public OrientedPlane(Map<VarType, Double> vec, double offset) {
    super();
    this.vec = Collections.unmodifiableMap(new HashMap<>(vec));
    this.offset = offset;
  }

  public static <VarType> OrientedPlane<VarType> minimum(VarType var, double min) {
    Map<VarType, Double> vec = new HashMap<>();
    vec.put(var, 1.0);
    return new OrientedPlane<>(vec, -min);
  }

  public Map<VarType, Double> getVec() {
    return vec;
  }

  public double getOffset() {
    return offset;
  }

  /**
   * Returns an equivalent plane whose vec has a l2 norm of 1.
   * 
   * @return
   */
  public OrientedPlane<VarType> normalize() {
    double l2Norm = MathUtils.l2Norm(vec);
    Map<VarType, Double> adjusted = MathUtils.adjustToMagnitude(vec, 1.0);
    return new OrientedPlane<>(adjusted, offset / l2Norm);
  }

  public double evaluate(Map<VarType, Double> context) {
    return MathUtils.dot(vec, context) + offset;
  }

  public String toString() {
    return "OrientedPlane(" + vec + " " + offset + ")";
  }

}
