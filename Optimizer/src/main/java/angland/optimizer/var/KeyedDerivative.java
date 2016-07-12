package angland.optimizer.var;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KeyedDerivative<VarKey> {

  private final IndexedKey<VarKey> key;
  private final double value;

  public KeyedDerivative(IndexedKey<VarKey> key, double value) {
    super();
    this.key = key;
    this.value = value;
  }

  public IndexedKey<VarKey> getKey() {
    return key;
  }

  public double getValue() {
    return value;
  }

  public static <VarKey> void printRelativeDist(Map<IndexedKey<VarKey>, Double> gradient) {
    Map<VarKey, List<Double>> grouped = new HashMap<>();
    for (Map.Entry<IndexedKey<VarKey>, Double> entry : gradient.entrySet()) {
      grouped.computeIfAbsent(entry.getKey().getVarKey(), x -> new ArrayList<>()).add(
          entry.getValue());
    }
    grouped.forEach((k, v) -> {
      System.out.println(k + ", count: " + v.stream().count() + ", average abs: "
          + v.stream().collect(Collectors.averagingDouble(x -> Math.abs(x))));
    });

  }
}
