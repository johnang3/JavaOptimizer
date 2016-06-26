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
public class ScalarValue<VarKey> {

  private final double value;
  private final Map<IndexedKey<VarKey>, Double> gradient;
  private final Map<IndexedKey<VarKey>, Double> context;

  ScalarValue(double value, Map<IndexedKey<VarKey>, Double> gradient,
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

  public ScalarValue<VarKey> plus(ScalarValue<VarKey> other) {
    double newVal = value + other.value();
    Map<IndexedKey<VarKey>, Double> newGrad = MathUtils.add(gradient, other.gradient);
    return new ScalarValue<>(newVal, newGrad, context);
  }

  public ScalarValue<VarKey> divide(ScalarValue<VarKey> other) {
    return times(other.power(-1));
  }

  /**
   * Returns e^x, where x is this calculation.
   * 
   * @return
   */
  public ScalarValue<VarKey> exp() {
    double newVal = Math.exp(value);
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    for (Map.Entry<IndexedKey<VarKey>, Double> e : gradient.entrySet()) {
      newGrad.put(e.getKey(), Math.exp(e.getValue()));
    }
    return new ScalarValue<>(newVal, newGrad, context);
  }

  @SuppressWarnings("unchecked")
  public static <VarKey> ScalarValue<VarKey> sum(ScalarValue<VarKey>... args) {
    if (args.length == 0) {
      throw new IllegalArgumentException("Cannot take the sum of zero elements.");
    }
    double newVal = 0;
    Map<IndexedKey<VarKey>, Double> newGrad = new HashMap<>();
    Consumer<Map.Entry<IndexedKey<VarKey>, Double>> accumulator =
        entry -> newGrad.merge(entry.getKey(), entry.getValue(), Double::sum);
    for (ScalarValue<VarKey> calc : args) {
      newVal += calc.value();
      calc.getGradient().entrySet().forEach(accumulator);
    }
    return new ScalarValue<>(newVal, newGrad, args[0].context);
  }

  public static class Builder<VarKey> {
    private double value = 0;
    private final Map<IndexedKey<VarKey>, Double> gradient;

    public Builder(int gradientVars) {
      gradient = new HashMap<>(gradientVars);
    }

    public double getValue() {
      return value;
    }

    public void setValue(double value) {
      this.value = value;
    }

    public void incrementValue(double value) {
      this.value += value;
    }

    public void increment(ScalarValue<VarKey> other) {
      this.value += other.value();
      other.getGradient().forEach((k, v) -> gradient.merge(k, v, Double::sum));
    }

    public Map<IndexedKey<VarKey>, Double> getGradient() {
      return gradient;
    }

    public ScalarValue<VarKey> build(Map<IndexedKey<VarKey>, Double> context) {
      return new ScalarValue<>(value, gradient, context);
    }

  }

  public ScalarValue<VarKey> times(ScalarValue<VarKey> other) {
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
    return new ScalarValue<>(newVal, newGrad, context);
  }

  public ScalarValue<VarKey> power(double exponent) {
    double newVal = Math.pow(value, exponent);
    Map<IndexedKey<VarKey>, Double> newGradient = new HashMap<>(gradient.size(), 1);
    gradient.forEach((k, v) -> newGradient.put(k, exponent * Math.pow(value, exponent - 1)));
    return new ScalarValue<>(newVal, newGradient, context);
  }

  public ArrayMatrixValue<VarKey> times(IMatrixValue<VarKey> matrix) {
    return ArrayMatrixValue.times(this, matrix);
  }

  public String toString() {
    return "Calculation(" + value + " " + gradient.toString() + ")";
  }

  public Map<IndexedKey<VarKey>, Double> getContext() {
    return context;
  }



}
