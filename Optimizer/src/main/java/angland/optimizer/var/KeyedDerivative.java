package angland.optimizer.var;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KeyedDerivative<VarKey> {

  private final ContextKey<VarKey> key;
  protected double value;

  public KeyedDerivative(ContextKey<VarKey> key, double value) {
    super();
    this.key = key;
    this.value = value;
  }

  public ContextKey<VarKey> getKey() {
    return key;
  }


  public double getValue() {
    return value;
  }

  public static <VarKey> void printRelativeDist(Map<? extends IndexedKey<VarKey>, Double> gradient) {
    System.out.println("Printing derivative dist:");
    Map<VarKey, List<Double>> grouped = new HashMap<>();
    for (Map.Entry<? extends IndexedKey<VarKey>, Double> entry : gradient.entrySet()) {
      grouped.computeIfAbsent(entry.getKey().getVarKey(), x -> new ArrayList<>()).add(
          entry.getValue());
    }
    grouped.forEach((k, v) -> {
      System.out.println(k + ", count: " + v.stream().count() + ", average abs: "
          + v.stream().collect(Collectors.averagingDouble(x -> Math.abs(x))));
    });

  }

  @Override
  public int hashCode() {
    return key.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) return false;
    if (!(other instanceof KeyedDerivative)) {
      return false;
    }
    @SuppressWarnings("rawtypes")
    KeyedDerivative casted = (KeyedDerivative) other;
    return key.equals(casted.key);
  }

}
