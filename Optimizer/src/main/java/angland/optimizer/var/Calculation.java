package angland.optimizer.var;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import angland.optimizer.vec.MathUtils;

/**
 * @author John Angland
 *
 * @param <VarKey>
 */
public class Calculation<VarKey> {

  private final double value;
  private final Map<IndexedKey<VarKey>, Double> gradient;
  private final Map<IndexedKey<VarKey>, Double> context;

  Calculation(double value, Map<IndexedKey<VarKey>, Double> gradient,
      Map<IndexedKey<VarKey>, Double> context) {
    super();
    this.value = value;
    this.gradient = Collections.unmodifiableMap(gradient);
    this.context = Collections.unmodifiableMap(context);
  }

  public double value() {
    return value;
  }

  public double d(IndexedKey<VarKey> v) {
    return gradient.getOrDefault(v, 0.0);
  }

  public Map<IndexedKey<VarKey>, Double> getGradient() {
    return gradient;
  }

  public Calculation<VarKey> plus(Calculation<VarKey> other) {
    double newVal = value + other.value();
    Map<IndexedKey<VarKey>, Double> newGrad = MathUtils.add(gradient, other.gradient);
    return new Calculation<>(newVal, newGrad, context);
  }

  @SuppressWarnings("unchecked")
  public static <VarKey> Calculation<VarKey> sum(Calculation<VarKey>... args) {
    if (args.length == 0) {
      throw new IllegalArgumentException("Cannot take the sum of zero elements.");
    }
    double newVal = 0;
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    Consumer<Map.Entry<IndexedKey<VarKey>, Double>> accumulator =
        entry -> newGrad.merge(entry.getKey(), entry.getValue(), Double::sum);
    for (Calculation<VarKey> calc : args) {
      newVal += calc.value();
      calc.getGradient().entrySet().forEach(accumulator);
    }
    return new Calculation<>(newVal, newGrad, args[0].context);
  }

  public static class Builder<VarKey> {
    private double value = 0;
    private final Map<IndexedKey<VarKey>, Double> gradient = new HashMap<>();

    public double getValue() {
      return value;
    }

    public void setValue(double value) {
      this.value = value;
    }

    public void incrementValue(double value) {
      this.value += value;
    }

    public void increment(Calculation<VarKey> other) {
      this.value += other.value();
      other.getGradient().forEach((k, v) -> gradient.merge(k, v, Double::sum));
    }

    public Map<IndexedKey<VarKey>, Double> getGradient() {
      return gradient;
    }

    public Calculation<VarKey> build(Map<IndexedKey<VarKey>, Double> context) {
      return new Calculation<>(value, gradient, context);
    }

  }

  public Calculation<VarKey> times(Calculation<VarKey> other) {
    double newVal = value * other.value();
    Map<IndexedKey<VarKey>, Double> newGrad =
        new HashMap<>(gradient.size() + other.gradient.size(), 1);
    for (Map.Entry<IndexedKey<VarKey>, Double> entry : this.gradient.entrySet()) {
      if (entry.getValue() == 0) continue;
      newGrad.merge(entry.getKey(), entry.getValue() * other.value(), Double::sum);
    }
    for (Map.Entry<IndexedKey<VarKey>, Double> entry : other.getGradient().entrySet()) {
      if (entry.getValue() == 0) continue;
      newGrad.merge(entry.getKey(), entry.getValue() * this.value(), Double::sum);
    }
    return new Calculation<>(newVal, newGrad, context);
  }

  public Calculation<VarKey> power(double exponent) {
    double newVal = Math.pow(value, exponent);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, exponent * Math.pow(value, exponent - 1)));
    return new Calculation<>(newVal, newGradient, context);
  }

  public MatrixValue<VarKey> times(MatrixValue<VarKey> matrix) {
    return MatrixValue.times(this, matrix);
  }

  public String toString() {
    return "Calculation(" + value + " " + gradient.toString() + ")";
  }

  public Map<IndexedKey<VarKey>, Double> getContext() {
    return context;
  }



}
